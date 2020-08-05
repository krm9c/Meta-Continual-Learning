import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time, copy
import gc

from collections import OrderedDict


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



# # Sanity Check
# if torch.cuda.is_available():
#     device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
#     print("Running on the GPU")
# else:
#     device = torch.device("cpu")
#     print("Running on the CPU")


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


# The lists for the two stuff
result_array_previous = []
result_array_current  = []
Experience_replay_x = []
Experience_replay_y = []


import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

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
            # x = x.to(device)
            predictions = self.model_g(self.model_h(x))
            temp = predictions.cpu().detach().numpy()
            total_yhat  = np.concatenate([total_yhat, temp])
            temp = sample['y'].detach().numpy()
            total_y  = np.concatenate([total_y, temp])
        return total_yhat, total_y

    def return_score(self,tasks):
        running_average = 0.0
        all_errors = []
        for i in range(tasks):
            # dataset = dataset_return(i)
            # dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
            # yhat, y = self.evaluate_model(dataloaders)
            error_value = self.return_score_current(i)
            # print(error_value, \
            # round(mean_squared_error(yhat, y),4))
            running_average += error_value
            all_errors.append(error_value)
        return (running_average/float(tasks)), all_errors

    def return_score_current(self,task_id):
        dataset = dataset_return(task_id)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        yhat, y = self.evaluate_model(dataloaders)
        return round(mean_squared_error(yhat, y),4)


    def copy_model(self,model_1):
        D_in = 3
        H = 1000
        D_out = 1000
        import copy
        # Model buffer
        model_buffer = copy.deepcopy(model_1)
        # model_buffer.to(device)
        opt2 = torch.optim.Adam(model_buffer.parameters(), lr = 0.0001) # get new optimiser
        return model_buffer, opt2


    def current_compensate(self, x, y, criterion):
        # x, y = x.to(device), y.to(device)
        # J(k)+ (V(k+1)-V(k)) can be split into two updates one 
        # for the current cost term 
        # and one for the future cost term
        # Update the model once for the current cost J(k)
        y_pred = self.model_g(self.model_h(x))
        J_k = criterion(y_pred, y)
        self.optimizer.zero_grad()
        J_k.backward(create_graph = True)
        self.optimizer.step()
    
    def Cata_compensate(self, x, y, zeta, criterion_MSE, criterion):
        ## Compensate for catastropic forgetting.
        ## Use the model buffer to get the future cost
        ## Now, we need to approximate the future cost and update the model again.
        buffer_model, opt_buffer = self.copy_model(self.model_g)
        
        ### The future cost and the updates
        # x, y = x.to(device), y.to(device) 
        #  Run the markov chain and get the future cost
        for _ in range(zeta): 
            opt_buffer.zero_grad()
            y_pred = buffer_model(self.model_h(x))
            Cost_temp = criterion(y_pred, y)
            Cost_temp.backward(create_graph = True)
            opt_buffer.step()
            del y_pred

            temp = x.cpu() + torch.rand(x.shape)
            # temp = temp.to(device)
            y_pred = buffer_model(self.model_h(temp))
            Cost_temp = criterion(y_pred, y)
            Cost_temp.backward(create_graph = True)
            opt_buffer.step()
            del temp

        #V(k)
        y_pred = self.model_g(self.model_h(x) )
        V_k = criterion(y_pred, y)
        ##V(k+1)               
        y_pred_F = buffer_model(self.model_h(x)) 
        V_k_plus_1 = criterion(y_pred_F, y)
        # V(k+1)-V(k)
        Total_loss =  0.1*criterion_MSE(V_k_plus_1,V_k) + V_k
        self.optimizer.zero_grad()
        Total_loss.backward(create_graph = True)
        self.optimizer.step()

        del buffer_model
        del opt_buffer
        del x, y, y_pred, y_pred_F



    def backward(self, K, batch_size, zeta, dataloaders, dataloaders_exp, criterion):
        dataloader_iterator = iter(dataloaders_exp)
        criterion_MSE = torch.nn.MSELoss() 
        # Internal loop with samples across 
        for current_sample in dataloaders:
            for _ in range(K):
                try:
                    past_sample = next(dataloader_iterator)
                except StopIteration:
                    continue

                #### Learn on the new task
                x = current_sample['x'].float()
                y = current_sample['y'].float()
                self.current_compensate(x, y, criterion)

                x = past_sample['x'].float()
                y = past_sample['y'].float()
                self.Cata_compensate(x, y, zeta, criterion_MSE, criterion)

                torch.cuda.empty_cache()
                gc.collect()
        return 


total_runs   = 5
total_samples = 50
learning_rate = 1e-3
N = 8
kappa = 30
zeta = 5

print("The parameters", total_runs, total_samples, learning_rate, N, kappa, zeta)
CME = np.zeros([total_runs, total_samples])
CTE = np.zeros([total_runs, total_samples])  
TE  = np.zeros([total_runs, total_samples]) 

for runs in range(total_runs):
    torch.manual_seed(runs)
    model = Net(8, 3, 1000, 1000, learning_rate)
    # model.to(device)
    Experience_replay_x = []
    Experience_replay_y = []
    for  samp_num in range(total_samples):
        dataset = dataset_return(samp_num)
        dataloaders = DataLoader(dataset, batch_size=N, shuffle=True, num_workers=1)
        
        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].float()
            # Compensate for the older tasks
            # Start by updating the data-stream
            if np.random.random() > 0.5:
                Experience_replay_x.extend(x.detach().numpy())
                Experience_replay_y.extend(y.detach().numpy())

        if len(Experience_replay_x)>1000:
            idx = list(range(len(Experience_replay_x)))
            idx_test = list(np.random.choice(idx, 1000, replace = False).reshape([-1]))
            Experience_replay_x = [Experience_replay_x[id] for id in idx_test]
            Experience_replay_y = [Experience_replay_y[id] for id in idx_test]
            # print(len(Experience_replay_y), len(Experience_replay_x))

        # Readjustment with the experience data
        dataset_exp =  Continual_Dataset(data_x =  Experience_replay_x, data_y =  Experience_replay_y)
        dataloaders_exp = DataLoader(dataset_exp, batch_size = N, shuffle=True, num_workers=1)
        
        model.backward(K = kappa, batch_size = N, zeta = zeta, \
        dataloaders = dataloaders, dataloaders_exp = dataloaders_exp, criterion = torch.nn.MSELoss())
        i = runs


        with torch.no_grad():
            # Evaluation section
            CTE[i, samp_num] = model.return_score_current(samp_num)
            CME[i,samp_num], _ = model.return_score( (samp_num+1))
            print('Sample_number {}/{}'.format(samp_num, total_samples - 1),\
            "Cumulative error", CME[i, samp_num], "Current error", CTE[i,samp_num])  
    _, TE[i,:] = model.return_score( (samp_num+1)) 
    del model
    del Experience_replay_y
    del Experience_replay_x

print("I finished the loop")
t = np.arange(total_samples)
plt.plot(t, CME[i,:])
plt.xlabel('tasks')
plt.ylabel('Cumulative Error')
plt.savefig("/home/kraghavan/Projects/Continual/DCL/Result_paper/DCL_training_previous_tasks_2.png", dpi = 1200)
plt.figure()
plt.plot(t, CTE[i,:])
plt.xlabel('tasks')
plt.ylabel('Current Task Error')
plt.savefig("/home/kraghavan/Projects/Continual/DCL/Result_paper/DCL_training_current_tasks_2.png", dpi = 1200)
np.savetxt('/home/kraghavan/Projects/Continual/DCL/Result_paper/CME_DCL_2.csv',\
CME, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/DCL/Result_paper/CTE_DCL_2.csv',\
CTE, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/DCL/Result_paper/TE_DCL_2.csv',\
TE, delimiter = ',')
