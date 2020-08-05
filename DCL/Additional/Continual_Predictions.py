import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time, copy


from collections import OrderedDict
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def display_batch(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['x'].size(),
              sample_batched['y'].size())
        
def dataset_return(i):
    import pickle
    with open('/gpfs/jlse-fs0/users/kraghavan/Continual/Incremental_Sine.p', 'rb') as fp:
        data = pickle.load(fp)
    y, time, phase, amplitude, frequency = data['task'+str(i)]
    x = np.concatenate([phase, amplitude.reshape([-1,1]), frequency.reshape([-1,1]) ], axis = 1)
    dataset = Continual_Dataset(data_x = x, data_y = y)
    return dataset


from sklearn.metrics import r2_score, mean_squared_error
class Continual_Dataset(Dataset):
    def __init__(self, data_x, data_y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = pd.DataFrame(data = data_x)
        self.y = pd.DataFrame(data = data_y)
    
    # A function to define the length of the problem
    def __len__(self):
        return len(self.x)
    
    # A function to get samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x_ = self.x.iloc[idx, :].values
        y_ = self.y.iloc[idx, :].values
        sample = {'x': x_, 'y': y_}
        return sample

def copy_model(model_1, model_2):    
    model_2.load_state_dict(model_1.state_dict())
    return model_2

def evaluate_model(model, dataloader_eval):
    model.eval()
    total_y = np.zeros([0,1000])
    total_yhat = np.zeros([0,1000])
    for sample in dataloader_eval:
        x= sample['x'].float()
        predictions = model(x)
        temp = predictions.detach().numpy()
        total_yhat  = np.concatenate([total_yhat, temp])
        temp = sample['y'].detach().numpy()
        total_y  = np.concatenate([total_y, temp])
    return total_yhat, total_y


def return_score(total, model):
    running_average = 0.0
    for i in range(total):
        dataset = dataset_return(i)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        yhat, y = evaluate_model(model,dataloaders)
        running_average += round(mean_squared_error(yhat, y),4)
    return (running_average/float(total))
        





import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(torch.nn.Module):
    def __init__(self, N, D_in, H, D_out, learning_rate):
        super(Net, self).__init__()
        
        # Model g
        self.model_g = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))
        
        # Model h
        self.model_h = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_in))

        parameter_list = list(self.model_g.parameters()) + list(self.model_h.parameters())
        self.optimizer = torch.optim.Adam( parameter_list, lr=learning_rate)


    def evaluate_model(self, dataloader_eval):
        self.model_g.eval()
        self.model_h.eval()
        total_y = np.zeros([0,1000])
        total_yhat = np.zeros([0,1000])
        for sample in dataloader_eval:
            x= sample['x'].float()
            predictions = self.model_g(self.model_h(x))
            temp = predictions.detach().numpy()
            total_yhat  = np.concatenate([total_yhat, temp])
            temp = sample['y'].detach().numpy()
            total_y  = np.concatenate([total_y, temp])
        return total_yhat, total_y


    def return_score(self,total):
        running_average = 0.0
        all_errors = []
        for i in range(total):
            dataset = dataset_return(i)
            dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
            yhat, y = self.evaluate_model(dataloaders)
            error_value = round(mean_squared_error(yhat, y),4)
            running_average += error_value
            all_errors.append(error_value)
        return (running_average/float(total)), all_errors


    def return_score_current(self,i):
        dataset = dataset_return(i)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        yhat, y = self.evaluate_model(dataloaders)
        return round(mean_squared_error(yhat, y),4)


    def return_predictions(self,total):
        Experience_replay_y =[]
        Experience_replay_yhat = []

        for i in range(total):
            dataset = dataset_return(i)
            dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
            yhat, y = self.evaluate_model(dataloaders)
            index_return = np.random.randint(0, yhat.shape[0], [2])
            Experience_replay_yhat.extend( yhat[index_return,:])
            Experience_replay_y.extend( y[index_return,:])

        print(len(Experience_replay_y), len(Experience_replay_yhat))
        return np.array(Experience_replay_y), np.array(Experience_replay_yhat)
    

    def copy_model(self,model_1):
        D_in = 3
        H = 1000
        D_out = 1000
        import copy
        # Model buffer
        model_buffer = copy.deepcopy(model_1)
        opt2 = torch.optim.Adam(model_buffer.parameters(), lr = 0.001) # get new optimiser
        return model_buffer, opt2


    def backward(self, num_epochs, K, batch_size, dataloaders, dataloaders_exp, criterion):
        dataloader_iterator = iter(dataloaders_exp)
        # outer loop with respect to the number of epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
            idx =0
            # Internal loop with samples across the dataset
            for sample in dataloaders:
                idx+=1
                x = sample['x'].float()
                y = sample['y'].float()

                # J(k)+ (V(k+1)-V(k)) can be split into two updates one 
                # for the current cost term 
                # and one for the future cost term
                # Update the model once for the current cost J(k)
                y_pred = self.model_g(self.model_h(x))
                J_k = criterion(y,y_pred)
                self.optimizer.zero_grad()
                J_k.backward(create_graph = True)
                self.optimizer.step()


                for _ in range(K):
                    ## Use the model buffer to get the future cost
                    #  Now, we need to approximate the future cost and update the model again.
                    buffer_model, opt_buffer = self.copy_model(self.model_g)

                    ## The future cost and the updates
                    try:
                        sample = next(dataloader_iterator)
                    except StopIteration:
                        continue

                    x = sample['x'].float()
                    y = sample['y'].float() 

                    
                    #  Run the markov chain and get the future cost
                    for _ in range(5): 
                        opt_buffer.zero_grad()
                        y_pred = buffer_model(self.model_h(x))
                        Cost_temp = criterion(y_pred, y)
                        Cost_temp.backward(create_graph = True)
                        opt_buffer.step()

                        # temp = x + torch.rand(x.shape)*0.01
                        # y_pred = buffer_model(self.model_h(temp))
                        # Cost_temp = criterion(y_pred, y)
                        # Cost_temp.backward(create_graph = True)
                        # opt_buffer.step()

                    #V(k)
                    y_pred = self.model_g(self.model_h(x) )
                    V_k = criterion(y,y_pred)

                    ##V(k+1)               
                    y_pred_F = buffer_model(self.model_h(x)) 
                    V_k_plus_1 = criterion(y,y_pred_F)

                    # V(k+1)-V(k)
                    Total_loss = 0.001*criterion(V_k_plus_1, V_k)
                    self.optimizer.zero_grad()
                    Total_loss.backward(create_graph = True)
                    self.optimizer.step()
                running_loss+=  J_k
        # print('epoch number {}/{}'.format(epoch, num_epochs - 1),\
        # (running_loss/float(idx))  )        
        del buffer_model
        del opt_buffer
        return running_loss


reruns = 1
total_samples = 50
all_errors = np.zeros([reruns, total_samples])
Result_array_current   = np.zeros([reruns, total_samples])
Result_array_previous  = np.zeros([reruns, total_samples])
for i in range(reruns):
    # Change the see for each run 
    torch.manual_seed(i)
    # The lists for the two stuff
    result_array_previous = []
    result_array_current  = []
    Experience_replay_x = []
    Experience_replay_y = []
    model = Net(8, 3, 1000, 1000, 0.0001)
    import time
    star_time = time.time() 
    for  samp_num in range(total_samples):
        dataset = dataset_return(samp_num)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].float()
            # Compensate for the older tasks
            # Start by updating the data-stream
            if np.random.random() > 0.5:
                Experience_replay_x.extend(x.detach().numpy())
                Experience_replay_y.extend(y.detach().numpy())
                
        # Readjustment with the experience data
        dataset_exp =  Continual_Dataset(data_x =  Experience_replay_x, data_y =  Experience_replay_y)
        dataloaders_exp = DataLoader(dataset_exp, batch_size = 8, shuffle=True, num_workers=1)
        loss = model.backward(num_epochs = 20, K = (2*samp_num+1), batch_size = 8,\
        dataloaders = dataloaders, dataloaders_exp = dataloaders_exp, criterion = torch.nn.MSELoss())
        
        error, y, yhat = model.return_predictions(total_samples)


        print('Sample number {}/{}'.format(samp_num, total_samples-1),error, loss, \
         "Time elapsed for the current run",(time.time()-star_time))
 
#np.savetxt('/home/kraghavan/Projects/Continual/Incremental_Sine/Result_Reruns/result_DCL_y.csv',y)
#np.savetxt('/home/kraghavan/Projects/Continual/Incremental_Sine/Result_Reruns/result_DCL_yhat.csv',yhat)