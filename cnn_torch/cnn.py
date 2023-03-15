import os  
import numpy as np
import time
import uproot
import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, ReLU, Sigmoid, Sequential
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

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


train_dataset, val_dataset = torch.utils.data.random_split( MyDataset("../data/data.root", num_samples),
                                                            [0.8, 0.2],
                                                            generator=torch.Generator().manual_seed(123))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the model as a 2d cnn autoencoder
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.encoder = nn.Sequential(
#         nn.Conv2d(1, 4, 3, padding=1),
#         nn.ReLU(True),
#         nn.MaxPool2d(2, stride=2),
#         nn.Conv2d(4, 8, 3, padding=1),
#         nn.ReLU(True),
#         nn.MaxPool2d(2, stride=2),
#         nn.Conv2d(8, 16, 3, padding=1),
#         nn.ReLU(True),
#         nn.MaxPool2d(2, stride=2)
#         )
#         self.decoder = nn.Sequential(
#         nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
#         nn.ReLU(True),
#         nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
#         nn.ReLU(True),
#         nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
#         nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

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
    
                          

class Stacked(nn.Module):
    def __init__(self):
        super(Stacked, self).__init__()
        self.s = nn.Sequential(
        nn.Conv2d(1, 8, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(8, 16, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(16, 32, 5, padding=2),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(32, 1, 3, padding=1),
        nn.Sigmoid()
        )
        
    
    def forward(self, x):
        return self.s(x)


c1 = Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1) # replicator
c2 = Conv2d(in_channels=2, out_channels=2, kernel_size=3, padding=1, groups=2) # depthwise
c3 = Conv2d(in_channels=2, out_channels=4, kernel_size=1, padding=1, stride=2) # pointwise ?

c4 = Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1) # replicator
c5 = Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8) # depthwise
c6 = Conv2d(in_channels=8, out_channels=5, kernel_size=1, padding=1, stride=2) # pointwise ?

c7 = Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1) # replicator
c8 = Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10) # depthwise
c9 = Conv2d(in_channels=10, out_channels=5, kernel_size=1, padding=1, stride=2) # pointwise ?

c10 = Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1) # replicator
c11 = Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10) # depthwise
c12 = ConvTranspose2d(in_channels=10, out_channels=4, kernel_size=1, padding=1, stride=2) # pointwise ?

c13 = Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1) # replicator
c14 = Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, groups=8) # depthwise
c15 = ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=1, padding=1, stride=2, output_padding=1) # pointwise ?

c16 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1) # replicator
c17 = Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6) # depthwise
c18 = ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=1, padding=1, stride=2, output_padding=1) # pointwise ?

c19 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1) # replicator
c20 = Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, groups=6) # depthwise
c21 = Conv2d(in_channels=6, out_channels=1, kernel_size=1, padding=1) # pointwise ?

model = Sequential(
    c1, ReLU(),
    c2, ReLU(),
    c3, ReLU(),
    c4, ReLU(),
    c5, ReLU(),
    c6, ReLU(),
    c7, ReLU(),
    c8, ReLU(),
    c9, ReLU(),
    c10, ReLU(),
    c11, ReLU(),
    c12, ReLU(),
    c13, ReLU(),
    c14, ReLU(),
    c15, ReLU(),
    c16, ReLU(),
    c17, ReLU(),
    c18, ReLU(),
    c19, ReLU(),
    c20, ReLU(),
    c21, Sigmoid(),
)

class mymodel(nn.Module):
    def __init__(self) -> None:
       super().__init__()
       self.net = model

    def forward(self, x):
        x = self.net(x)
        return x

# Initialize early stopping counter
early_stop_counter = 0

# Set up the model, loss function, and optimizer
# Create an instance of the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Stacked().to(device)
model = UNet().to(device)
#loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# non empty accuracy
def accuracy(output, target, input):
    mask = (input > 0.01)
    output = output[mask] > 0.5
    target = target[mask] > 0.5
    return (output == target).float().mean()

best_val_loss = np.inf
patience = patience

for epoch in range(max_epochs):
    # Train the model
    print(f"Epoch {epoch+1} of {max_epochs}")
    print("-" * 10)
    model.train()
    train_loss = 0
    train_acc = 0
    for input, target in train_loader:
        input = input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.binary_cross_entropy(output, target, reduction="mean")
        loss.backward()
        train_loss += loss.item()
        train_acc += accuracy(output, target, input)
        optimizer.step()
    print(f"====> Epoch: {epoch+1} Average loss: {train_loss / (len(train_loader.dataset)/batch_size):.4f}  Accuracy: {train_acc / (len(train_loader.dataset)/batch_size):.4f}")

    # Evaluate the model
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            val_loss += F.binary_cross_entropy(output, target, reduction="mean")
            val_acc += accuracy(output, target, input)
    val_loss /= len(val_loader.dataset)/batch_size
    print(f"====> Validation set loss: {val_loss:.4f} Accuracy: {val_acc / (len(val_loader.dataset)/batch_size):.4f}")
    # Early stopping
    if val_loss > best_val_loss:
        early_stop_counter += 1
        if early_stop_counter == patience:
            print("Early stopping")
            break
    else:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "model_best_cnn.pt")
        early_stop_counter = 0

# calculate the AUC score on the test set using the val_loader
model.load_state_dict(torch.load('model_best_cnn.pt'))
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for input, target in val_loader:
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        mask = (input > 0.01)
        y_true.extend(target[mask].cpu().numpy())
        y_pred.extend(output[mask].cpu().numpy())

y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
print(f"AUC score: {roc_auc_score(y_true, y_pred)}")

# Save the model to onnx file format
# use cpu here
device = torch.device('cpu')
model = model.to(device)
x = torch.rand(1, 1, 72, 32)
dynamic_axes = {"input": {0: 'batch_size'}, "output": {0: 'batch_size'}}
torch.onnx.export(model, x, "unet_full_mse.onnx", input_names=["input"], output_names=["output"], dynamic_axes=dynamic_axes)



