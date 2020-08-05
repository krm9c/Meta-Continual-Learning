import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time, copy

# Sanity Check
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def display_batch(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['x'].size(),
              sample_batched['y'].size())
        
def dataset_return(i, flag = 'training'):
    if flag == 'training':
        import pickle
        with open('/gpfs/jlse-fs0/users/kraghavan/Continual/Incremental_Sine.p', 'rb') as fp:
            data = pickle.load(fp)
        y, time, phase, amplitude, frequency = data['task'+str(i)]
        x = np.concatenate([phase, amplitude.reshape([-1,1]), frequency.reshape([-1,1]) ], axis = 1)
        dataset = Continual_Dataset(data_x = x, data_y = y)
    else:
        import pickle
        # print("importing test data")
        with open('/gpfs/jlse-fs0/users/kraghavan/Continual/Incremental_Sine_test.p', 'rb') as fp:
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


def evaluate_model(model_theta, model_W, dataloader_eval):
    model_theta.eval()
    model_W.eval()
    total_y = np.zeros([0,1000])
    total_yhat = np.zeros([0,1000])
    for sample in dataloader_eval:
        x= sample['x'].float().to(device)
        predictions = model_W(model_theta(x)).cpu()
        temp = predictions.detach().numpy()
        total_yhat  = np.concatenate([total_yhat, temp])
        temp = sample['y'].detach().numpy()
        total_y  = np.concatenate([total_y, temp])
    return round(mean_squared_error(yhat, y),4

def return_score(total, model_theta, model_W):
    running_average = 0.0
    all_errors = []
    for i in range(total):
        dataset = dataset_return(i, flag = 232)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        error_value = evaluate_model(model_theta, model_W, dataloaders)
        running_average += error_value
        all_errors.append(error_value)
    return (running_average/float(total)), all_errors


def return_score_current(i, model_theta, model_W):
    dataset = dataset_return(i, flag = 232)
    dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    yhat, y = evaluate_model(model_theta, model_W, dataloaders)
    return round(mean_squared_error(yhat, y),4)
        


total_runs = 10
num_epochs = 1
total_samples = 50
learning_rate = 1e-3


for runs in range(total_runs):
    torch.manual_seed(runs)

    # The final code runs
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 8, 3, 100, 1000
    CME = np.zeros([total_runs, total_samples])
    CTE = np.zeros([total_runs, total_samples])  
    TE  = np.zeros([total_runs, total_samples])  

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    # Make the Representation Learning network
    model_theta = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_in))
    model_theta.to(device)

    # The lists for the two stuff
    Experience_replay_x = []
    Experience_replay_y = []
    # Loss criterion
    criterion = torch.nn.MSELoss()
    optimizer_theta = torch.optim.Adam( model_theta.parameters(), lr=learning_rate)
    # Meta Training 
    # The main working loop
    for  samp_num in range(total_samples):
        # Initialize all the dataloaders for the experience array and the other dataset
        dataset = dataset_return(samp_num)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        # Make the network
        model_W = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out))
        model_W.to(device)
        optimizer_W = torch.optim.Adam( model_W.parameters(), lr=learning_rate)
        # First the Meta Training
        for epoch in range(num_epochs):
            for sample in dataloaders:
                x = sample['x'].float()
                y = sample['y'].float()
                x, y = x.to(device), y.to(device)
                y_pred = model_W(model_theta(x)) 
                loss = criterion(y_pred, y) 
                optimizer_W.zero_grad()
                loss.backward(create_graph=True)
                optimizer_W.step()
            # I have kind of learned the classifier. 
            # Next, Learn the 
            for sample in dataloaders:
                x = sample['x'].float()
                y = sample['y'].float()
                x, y = x.to(device), y.to(device)
                y_pred = model_W(model_theta(x)) 
                loss = criterion(y_pred, y) 
                optimizer_theta.zero_grad()
                loss.backward(create_graph=True)
                optimizer_theta.step()

    print("Representation ready, Moving to the Meta testing phase")
    del model_W
    del optimizer_W
    # Meta Testing.
    # Make the network
    model_W = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))
    model_W.to(device)
    optimizer_W = torch.optim.Adam( model_W.parameters(), lr= learning_rate)

    # The main working loop
    for  samp_num in range(total_samples):
        # Initialize all the dataloaders for the experience array and the other dataset
        dataset = dataset_return(samp_num)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].float()
            x, y = x.to(device), y.to(device)
            y_pred = model_W(model_theta(x)) 
            loss = criterion(y_pred, y) 
            optimizer_W.zero_grad()
            loss.backward(create_graph=True)
            optimizer_W.step()

        # I am going to add a fine tuning step because, the thing blows up otherwise.
        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].float()
            x, y = x.to(device), y.to(device)
            y_pred = model_W(model_theta(x)) 
            loss = criterion(y_pred, y) 
            optimizer_theta.zero_grad()
            loss.backward(create_graph=True)
            optimizer_theta.step()


        i = runs
        print(i, samp_num)

        ##   Evaluation section
        CTE[i, samp_num] =  return_score_current(samp_num, model_theta, model_W)
        CME[i,samp_num], _ = return_score( (samp_num+1), model_theta, model_W)

        print('Sample_number {}/{}'.format(samp_num, total_samples - 1), \
        "Cumulative error", CME[i, samp_num], "previous error", CTE[i,samp_num])   

    # Print and display the final things.
    t = np.arange(total_samples)
    _, TE[i,:] = return_score( (total_samples), model_theta, model_W)
    del model_W
    del model_theta

    plt.plot(t, CME[i,:])
    plt.xlabel('tasks')
    plt.ylabel('Cumulative Error')
    plt.savefig("/home/kraghavan/Projects/Continual/CML/Results_paper/CML_training_previous_tasks.png", dpi = 1200)
    plt.figure()
    plt.plot(t, CTE[i,:])
    plt.xlabel('tasks')
    plt.ylabel('Current Task Error')
    plt.savefig("/home/kraghavan/Projects/Continual/CML/Results_paper/CML_training_current_tasks.png", dpi = 1200)
    np.savetxt('/home/kraghavan/Projects/Continual/CML/Results_paper/CME_CML.csv',\
    CME, delimiter = ',')
    np.savetxt('/home/kraghavan/Projects/Continual/CML/Results_paper/CTE_CML.csv',\
    CTE, delimiter = ',')
    np.savetxt('/home/kraghavan/Projects/Continual/CML/Results_paper/TE_CML.csv',\
    TE, delimiter = ',')
