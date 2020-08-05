# PyTorch libraries and modules
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d
from torch.nn import Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time, copy
from sklearn.metrics import accuracy_score
import gc

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
def display_batch(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['x'].size(),
              sample_batched['y'].size())


# Sanity Check
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



import torchvision
from PIL.Image import LANCZOS
OMNI = torchvision.datasets.Omniglot(root="../data", download=True, transform=transforms.Compose([
                                                    transforms.Resize(28, interpolation=LANCZOS),
                                                    transforms.ToTensor(),
                                                    lambda x: 1.0 - x,
                                                ]))



def dataset_make_omniglot(task_id):
    import numpy as np 
    temp_list = []
    temp_label = []
    prev = 0
    data = {}
    temp = (task_id)*20
    idx = np.random.randint(temp, temp+20, 20)
    temp_list = []
    labels_list = []
    for element in idx:
        images, labels = OMNI[element]
        temp_list.append(images)
        labels_list.append(labels)
    images = torch.stack(temp_list, dim=0)
    labels = torch.from_numpy(  np.array(labels_list).reshape([20]) ) 
    s = list(range(15, 20))
    idx_test = np.random.choice(s, 100, replace = True)
    # print(idx_test)
    s = list(range(0, 15))
    idx_train = np.random.choice(s, 15, replace = False)
    # print(idx_train)
    x_ = images[idx_test,:,:,:]
    y_ = labels[idx_test]
    x = images[idx_train,:,:,:]
    y = labels[idx_train]
    # print(x.shape, y.shape, x_.shape, y_.shape)  
    return x, y, x_, y_ 

        
def dataset_return(task_id, flag = 'training'):
    ## The task id starts from zero!! Remember that
    x, y, x_test, y_test = dataset_make_omniglot(task_id)
    if flag == 'training':
        dataset = Continual_Dataset(data_x = x, data_y = y)
        return dataset
    else:
        dataset = Continual_Dataset(data_x = x_test, data_y = y_test)
        return dataset


def dataset_make_omniglot_exp(task_id):
    import numpy as np 
    temp_list = []
    temp_label = []
    prev = 0
    data = {}
    idx = range( (task_id)*20 )
    temp_list = []
    labels_list = []
    for element in idx:
        images, labels = OMNI[element]
        temp_list.append(images)
        labels_list.append(labels)
    images = torch.stack(temp_list, dim=0)
    labels = torch.from_numpy(  np.array(labels_list).reshape([len(temp_list)]) ) 

    if images.shape[0]>200:
        s = list(range(images.shape[0]))
        idx_test = np.random.choice(s, 200, replace = False)
        x = images[idx_test,:,:,:]
        y = labels[idx_test]
    else:
        s = list(range(images.shape[0]))
        idx_test = np.random.choice(s, 200, replace = True)
        x = images[idx_test,:,:,:]
        y = labels[idx_test]
    return x, y

def dataset_return_exp(task_id, flag = 'training'):
    ## The task id starts from zero!! Remember that
    if task_id == 0:
        return dataset_return(task_id, flag = 'training')
    else:
        x, y = dataset_make_omniglot_exp(task_id)
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
        self.x = data_x
        self.y = data_y
        self.transform = transforms.Compose([transforms.ToTensor()])  # you can add to the list all the transformations you need.
    
    # A function to define the length of the problem
    def __len__(self):
        return len(self.x)
    
    # A function to get samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        x_ = self.x[idx,:,:,:]
        y_ = self.y[idx]
    
        sample = {'x': x_, 'y': y_}
        return sample




# The lists for the two stuff
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


    def return_score(self, tasks):
        running_average = 0.0
        all_errors =[]
        for i in range(tasks):
            error_value = self.return_score_current(i) 
            running_average += error_value
            all_errors.append(error_value)
        return (running_average/float(tasks)), all_errors


    def return_score_current(self, task_id):
        self.model_h.eval()
        self.model_g.eval()
        dataset = dataset_return(task_id)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        total = 0
        score = 0.0
        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].long()
            x = x.reshape([-1,784])
            # print(x.shape, y.shape)        
            x, y = x.to(device), y.to(device)
            y_pred = self.model_g(self.model_h(x)) 
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            score += (predicted == y).sum().item()
        return  (score/float(total))


    def copy_model(self,model_1):
        import copy
        # Model buffer
        model_buffer = copy.deepcopy(model_1)
        model_buffer = model_buffer.to(device)
        opt2 = torch.optim.Adam(model_buffer.parameters(), lr = 0.0001) # get new optimiser
        return model_buffer, opt2


    def current_compensate(self, x, y, criterion):
        x, y = x.to(device), y.to(device)
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
        x, y = x.to(device), y.to(device) 
        #  Run the markov chain and get the future cost
        for _ in range(zeta): 
            opt_buffer.zero_grad()
            y_pred = buffer_model(self.model_h(x))
            Cost_temp = criterion(y_pred, y)
            Cost_temp.backward(create_graph = True)
            opt_buffer.step()
            del y_pred

            temp = x.cpu() + torch.rand(x.shape)
            temp = temp.to(device)
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
                y = current_sample['y'].long()
                x = x.reshape([-1,784])
                self.current_compensate(x, y, criterion)

                x = past_sample['x'].float()
                y = past_sample['y'].long()
                x = x.reshape([-1,784])
                self.Cata_compensate(x, y, zeta, criterion_MSE, criterion)
                torch.cuda.empty_cache()
                gc.collect()
        return 


total_runs   = 50
total_samples = 10
learning_rate = 1e-3
N = 4
kappa = 30
zeta = 5
CME = np.zeros([total_runs, total_samples])
CTE = np.zeros([total_runs, total_samples])  
TE  = np.zeros([total_runs, total_samples]) 
 
print("The parameters", total_runs, total_samples, learning_rate, N, kappa, zeta)
for i in range(total_runs):
    torch.manual_seed(i)
    # The main working loop
    model = Net(N, 784, 100, total_samples, learning_rate)
    model = model.to(device)
    import time
    for  samp_num in range(total_samples):
        dataset_current = dataset_return((samp_num))
        dataloaders = DataLoader(dataset_current, batch_size = N, shuffle=True, num_workers=1)

        dataset = dataset_return_exp((samp_num))
        dataloaders_experience = DataLoader(dataset, batch_size = N, shuffle=True, num_workers=1)

        star_time = time.time() 
        loss = model.backward(K = kappa, batch_size = N, zeta = zeta, \
        dataloaders = dataloaders, dataloaders_exp = dataloaders_experience, criterion =  torch.nn.CrossEntropyLoss())
        
        del dataloaders_experience
        del dataloaders
        with torch.no_grad():
            # Evaluation section
            CTE[i, samp_num]    = model.return_score_current(samp_num)
            CME[i, samp_num], _ = model.return_score(samp_num+1)
            print('Sample_number {}/{}'.format(samp_num, total_samples - 1),\
            "CME", CME[i, samp_num], "CTE", CTE[i,samp_num])  

    with torch.no_grad():            
        _, TE[i,:] = model.return_score( (samp_num+1)) 
    del model
print("I am done with the loop")

t = np.arange(total_samples)
plt.plot(t, CME[i,:])
plt.xlabel('tasks')
plt.ylabel('Cumulative Error')
plt.savefig("/home/kraghavan/Projects/Continual/DCL/Result_paper/DCL_training_previous_tasks_OMNI_2.png", dpi = 1200)
plt.figure()
plt.plot(t, CTE[i,:])
plt.xlabel('tasks')
plt.ylabel('Current Task Error')
plt.savefig("/home/kraghavan/Projects/Continual/DCL/Result_paper/DCL_training_current_tasks_OMNI_2.png", dpi = 1200)
np.savetxt('/home/kraghavan/Projects/Continual/DCL/Result_paper/CME_DCL_OMNI_2.csv',\
CME, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/DCL/Result_paper/CTE_DCL_OMNI_2.csv',\
CTE, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/DCL/Result_paper/TE_DCL_OMNI_2.csv',\
TE, delimiter = ',')
