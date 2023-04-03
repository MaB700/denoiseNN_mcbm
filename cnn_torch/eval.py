import os  
import numpy as np
import time
import uproot
import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, ReLU, Sigmoid, Sequential
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
# Set up some hyperparameters
num_samples = None # sum of train and val
batch_size = 256
max_epochs = 150
patience = 3

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, path, samples):
    self.x = None
    self.y = None
    with uproot.open(path) as file:
        self.x = np.reshape(np.array(file["train"]["time"].array(entry_stop=samples)), (-1, 1, 72, 32))    
        self.y = np.reshape(np.array(file["train"]["tar"].array(entry_stop=samples)), (-1, 1, 72, 32))

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


test_dataset = MyDataset("../data/data_test.root", num_samples)
                                                            

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

class UNet(nn.Module):
    def __init__(self, cl=[8, 8, 16, 32], bnorm=True):
        super(UNet, self).__init__()
        
        self.c1 = nn.Sequential(
            nn.Conv2d(1, cl[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[0]) if bnorm else nn.Identity()
        )
        self.p1 = nn.MaxPool2d(kernel_size=2)
        
        self.c2 = nn.Sequential(
            nn.Conv2d(cl[0], cl[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[1]) if bnorm else nn.Identity()
        )
        self.p2 = nn.MaxPool2d(kernel_size=2)
        
        self.c3 = nn.Sequential(
            nn.Conv2d(cl[1], cl[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[2]) if bnorm else nn.Identity()
        )
        self.p3 = nn.MaxPool2d(kernel_size=2)
        
        self.mid = nn.Sequential(
            nn.Conv2d(cl[2], cl[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[3]) if bnorm else nn.Identity()
        )
        
        self.u10 = nn.ConvTranspose2d(cl[3], cl[3], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c10 = nn.Sequential(
            nn.Conv2d(cl[3]+cl[2], cl[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[2]) if bnorm else nn.Identity()
        )
        
        self.u11 = nn.ConvTranspose2d(cl[2], cl[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c11 = nn.Sequential(
            nn.Conv2d(cl[2]+cl[1], cl[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[1]) if bnorm else nn.Identity()
        )
        
        self.u12 = nn.ConvTranspose2d(cl[1], cl[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c12 = nn.Sequential(
            nn.Conv2d(cl[1]+cl[0], cl[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[0]) if bnorm else nn.Identity()
        )
        
        self.c13 = nn.Sequential(
            nn.Conv2d(cl[0]+1, cl[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cl[0]) if bnorm else nn.Identity(),
            nn.Conv2d(cl[0], 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        mid = self.mid(p3)
        u10 = self.u10(mid)
        c10 = self.c10(torch.cat([u10, c3], dim=1))
        u11 = self.u11(c10)
        c11 = self.c11(torch.cat([u11, c2], dim=1))
        u12 = self.u12(c11)
        c12 = self.c12(torch.cat([u12, c1], dim=1))
        c13 = self.c13(torch.cat([c12, x], dim=1))
        return c13
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

model.load_state_dict(torch.load('model_best_cnn.pt'))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for input, target in test_loader:
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        mask = (input > 0.01)
        y_true.extend(target[mask].cpu().numpy())
        y_pred.extend(output[mask].cpu().numpy())

y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
print(f"AUC score: {roc_auc_score(y_true, y_pred)}")

# calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred.round())
print(cm)

# plot the response values for signal and background

plt.hist(y_pred[y_true == 1], bins=100, range=(0, 1), histtype='step', label='true hit')
plt.hist(y_pred[y_true == 0], bins=100, range=(0, 1), histtype='step', label='noise hit')
plt.xlabel('NN response')
plt.ylabel('Entries')
plt.yscale('log')
# move legend slightly to the left
plt.legend(loc='upper center')
plt.show()


