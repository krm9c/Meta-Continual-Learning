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



def evaluate_model(model_NM, model_RN, model_PLN, dataloader_eval):
    model_NM.eval()
    model_PLN.eval()
    model_RN.eval()
    total_y = np.zeros([0,1000])
    total_yhat = np.zeros([0,1000])
    for sample in dataloader_eval:
        x= sample['x'].float().to(device)
        predictions =forward(model_NM, model_RN, model_PLN,x).cpu()
        temp = predictions.detach().numpy()
        total_yhat  = np.concatenate([total_yhat, temp])
        temp = sample['y'].detach().numpy()
        total_y  = np.concatenate([total_y, temp])
    return total_yhat, total_y

def return_score(total, model_NM, model_RN, model_PLN):
    running_average = 0.0
    all_errors = []
    for i in range(total):
        dataset = dataset_return(i, flag = '232')
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
        yhat, y = evaluate_model(model_NM, model_RN, model_PLN, dataloaders)
        errors = round(mean_squared_error(yhat, y),4)
        all_errors.append(errors)
        running_average += errors
    return (running_average/float(total)), all_errors


def return_score_current(i, model_NM, model_RN, model_PLN):
    dataset = dataset_return(i, flag = '232')
    dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)
    yhat, y = evaluate_model(model_NM, model_RN, model_PLN, dataloaders)
    return round(mean_squared_error(yhat, y),4)
        

total_runs = 10
num_epochs = 30
total_samples = 50
learning_rate  = 1e-3
CME= np.zeros([total_runs, total_samples])
CTE= np.zeros([total_runs, total_samples])  
TE= np.zeros([total_runs, total_samples]) 

for i in range(total_runs):
    # The final code runs
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 8, 3, 1000, 1000
    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    # Make the Representation Learning network
    model_PLN = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out))
    model_PLN.to(device)

    # Make the Representation Learning network
    model_RN = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_in), 
        torch.nn.Sigmoid())

    model_RN.to(device)

    model_NM = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_in), 
        torch.nn.Sigmoid() )
    model_NM.to(device)


    def forward(model_NM, model_RN, model_PLN, x):
        x1 = model_NM(x)
        x2 = model_RN(x)
        x = x1*x2
        return model_PLN(x)


    # The lists for the two stuff
    result_array_previous = []
    result_array_current  = []
    Experience_replay_x = []
    Experience_replay_y = []

    # Loss criterion
    criterion = torch.nn.MSELoss()
    optimizer_NM = torch.optim.Adam( model_NM.parameters(),   lr=learning_rate)
    list_param = list(model_PLN.parameters())+ list(model_RN.parameters())
    optimizer_PLN  = torch.optim.Adam(list_param,  lr=learning_rate)
    # Meta Training 
    # The main working loop
    for  samp_num in range(total_samples):
        dataset = dataset_return(samp_num)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].float()
            index_return = np.random.randint(0,x.shape[0], [5])
            Experience_replay_x.extend(x[index_return, :].detach().numpy())
            Experience_replay_y.extend(y[index_return, :].detach().numpy())

        dataset_experience =  Continual_Dataset(data_x =  Experience_replay_x, data_y =  Experience_replay_y)
        dataloaders_experience = DataLoader(dataset_experience, batch_size=25, shuffle=True, num_workers=1)

        for epoch in range(num_epochs):
            # From task 
            for sample in dataloaders:
                x = sample['x'].float()
                y = sample['y'].float()
                x, y = x.to(device), y.to(device)
                y_pred = forward(model_NM, model_RN, model_PLN, x)
                loss = criterion(y_pred, y) 
                optimizer_PLN.zero_grad()
                loss.backward(create_graph=True)
                optimizer_PLN.step()
            # I have kind of learned the classifier. 
            # Next, Learn the Neuromodulatory module
            for sample in dataloaders_experience:
                x = sample['x'].float()
                y = sample['y'].float()
                x, y = x.to(device), y.to(device)
                y_pred = forward(model_NM, model_RN, model_PLN, x)
                loss = criterion(y_pred, y) 
                optimizer_NM.zero_grad()
                optimizer_PLN.zero_grad()
                loss.backward(create_graph=True)
                optimizer_NM.step()
                optimizer_PLN.step()

    print('Meta training done, Meta testing started',)
    # Meta Testing loop
    for  samp_num in range(total_samples):
        # Initialize all the dataloaders for the experience array and the other dataset
        dataset = dataset_return(samp_num)
        dataloaders = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=1)

        # First the Meta Training
        for sample in dataloaders:
            x = sample['x'].float()
            y = sample['y'].float()
            x, y = x.to(device), y.to(device)
            y_pred = forward(model_NM, model_RN, model_PLN, x)
            loss = criterion(y_pred, y) 
            optimizer_PLN.zero_grad()
            loss.backward(create_graph=True)
            optimizer_PLN.step()

        # Evaluation section
        CME[i, samp_num] = return_score_current(samp_num, model_NM, model_RN, model_PLN)
        CTE[i, samp_num], _ = return_score( (samp_num+1), model_NM, model_RN, model_PLN)
        print('Sample_number {}/{}'.format(samp_num, total_samples - 1),\
            "Cumulative error", CME[i, samp_num], "Current error", CTE[i,samp_num])      
            

    _, TE[i,:]= return_score( (samp_num+1), model_NM, model_RN, model_PLN)

    del model_NM
    del model_PLN
    del model_RN



t = np.arange(total_samples)
# Print and display the final things.
plt.plot(t, CME[i,:])
plt.xlabel('tasks')
plt.ylabel('Cumulative Error')
plt.savefig("/home/kraghavan/Projects/Continual/ANML/Result_paper/ANML_training_previous_tasks.png", dpi = 1200)
plt.figure()
plt.plot(t, CTE[i,:])
plt.xlabel('tasks')
plt.ylabel('Current Task Error')
plt.savefig("/home/kraghavan/Projects/Continual/ANML/Result_paper/ANML_training_current_tasks.png", dpi = 1200)
np.savetxt('/home/kraghavan/Projects/Continual/ANML/Result_paper/CME_ANML.csv',\
CME, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/ANML/Result_paper/CTE_ANML.csv',\
CTE, delimiter = ',')
np.savetxt('/home/kraghavan/Projects/Continual/ANML/Result_paper/TE_ANML.csv',\
TE, delimiter = ',')

