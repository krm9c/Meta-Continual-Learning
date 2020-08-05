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
    idx = range( (task_id+1)*20 )
    temp_list = []
    labels_list = []
    for element in idx:
        images, labels = OMNI[element]
        temp_list.append(images)
        labels_list.append(labels)
    images = torch.stack(temp_list, dim=0)
    labels = torch.from_numpy(  np.array(labels_list).reshape([len(temp_list)]) ) 
    return images, labels

def dataset_return_exp(task_id, flag = 'training'):
    ## The task id starts from zero!! Remember that
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


def return_score(total, model):
    running_average = 0.0
    all_errors =[]
    for i in range(total):
        error_value = return_score_current(i, model) 
        running_average += error_value
        all_errors.append(error_value)
    return (running_average/float(total)), all_errors


def return_score_current(task_id, model):
    model.eval()
    dataset = dataset_return((task_id), flag = 'testing')
    dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    idx = 0
    score = 0.0
    for sample in dataloaders:
        x = sample['x'].float()
        y = sample['y'].long()
        x = x.reshape([-1,784])
        # idx = idx+1
        # print(x.shape, y.shape)        
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        # score+= accuracy(y.cpu(),y_pred.cpu())
        _, predicted = torch.max(y_pred.data, 1)
        idx += y.size(0)
        score += (predicted == y).sum().item()
    return  (100*(score/idx))

total_runs   = 10
total_samples = 50
learning_rate = 1e-3
num_epochs = 1
N = 8
CME = np.zeros([total_runs, total_samples])
CTE = np.zeros([total_runs, total_samples])  
TE  = np.zeros([total_runs, total_samples]) 
for i in range(total_runs):

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 4, 784, 100, total_samples

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Make the network
    model_k = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )
    model_k.to(device)

    # Loss criterion
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_meta = torch.optim.Adam( model_k.parameters(), lr=learning_rate)

    # The main working loop
    for samp_num in range(total_samples):
        running_loss = 0.0
        score = 0.0
        total = 0
        # Training on the current task
        dataset = dataset_return_exp(samp_num)
        dataloaders = DataLoader(dataset, batch_size = N, shuffle=True, num_workers=1)

        for epoch in range(num_epochs):
            idx = 0
            for sample in dataloaders:
                idx = idx+1
                x = sample['x'].float()
                y = sample['y'].long()
                x = x.reshape([-1,784])
                # print(x.shape, y.shape)        
                x, y = x.to(device), y.to(device)
                y_pred = model_k(x) 
                loss = criterion(y_pred, y) 
                optimizer_meta.zero_grad()
                loss.backward(create_graph=True)
                optimizer_meta.step()

            #     running_loss += loss.item()
            #     _, predicted = torch.max(y_pred, 1)
            #     total += y.size(0)
            #     score += (predicted == y).sum().item()
            # # Experience replay driven training
            # epoch_score = score/total  
            # epoch_loss = running_loss/idx

        ##   Evaluation section
        
        CTE[i, samp_num] = return_score_current(samp_num, model_k)
        CME[i,samp_num], _ = return_score(samp_num+1, model_k)
        print(i, 'Sample_number {}/{}'.format(samp_num, total_samples - 1),\
            "Cumulative error", CME[i, samp_num], "Current error", CTE[i,samp_num])      
    
    # Print and display the final things.
    t = np.arange(total_samples)
    _, TE[i,:]= return_score( (total_samples), model_k)
    del model_k


# plt.plot(t, CME[i,:])
# plt.xlabel('tasks')
# plt.ylabel('Cumulative Error')
# plt.savefig("/home/kraghavan/Projects/Continual/ER/Result_paper/ER_training_previous_tasks_OMNI.png", dpi = 1200)
# plt.figure()
# plt.plot(t, CTE[i,:])
# plt.xlabel('tasks')
# plt.ylabel('Current Task Error')
# plt.savefig("/home/kraghavan/Projects/Continual/ER/Result_paper/ER_training_current_tasks_OMNI.png", dpi = 1200)


np.savetxt('/home/kraghavan/Projects/Continual/ER/Result_paper/CME_ER_OMNI.csv',\
CME, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/ER/Result_paper/CTE_ER_OMNI.csv',\
CTE, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/ER/Result_paper/TE_ER_OMNI.csv',\
TE, delimiter = ',')

