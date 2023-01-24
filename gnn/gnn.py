import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from helpers import *
from helpers_custom import *
import mcbm_dataset

import wandb
wandb.init(entity="mabeyer", project="GNN_denoise") # , mode='disabled'


reload = False
num_samples = None
num_samples_test = None
node_distance = 5

batch_size = 32 # FIXME: 256
epochs = 100
es_patience = 5

print(  "num_samples: ", num_samples, "num_samples_test: ", num_samples_test, 
        "node_distance: ", node_distance, "batch_size: ", batch_size, 
        "epochs: ", epochs, "es_patience: ", es_patience)

# train_dataset, val_dataset = \
#     torch.utils.data.random_split(CreateGraphDataset("../data.root:train", num_samples, dist = node_distance),
#                                                     [0.8, 0.2],
#                                                     generator=torch.Generator().manual_seed(123))

train_dataset, val_dataset = \
    torch.utils.data.random_split(mcbm_dataset.MyDataset(dataset="train", N = num_samples, reload=reload),
                                                    [0.8, 0.2],
                                                    generator=torch.Generator().manual_seed(123))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = customGNN().to(device)
# model = Net(train_dataset[0], 32).to(device)
model = model.to(torch.float)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_step():
    model.train()
    all_loss = 0
    all_acc = 0
    i = 0.0
    for data in train_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(output, data.y, reduction="mean")
        all_loss += loss.item()
        all_acc += accuracy(data.y, output)
        i += 1.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return all_loss/i, all_acc/i

def evaluate(loader):
    model.eval()
    all_loss = 0
    all_acc = 0
    i = 0.0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(output, data.y, reduction="mean")
        all_loss += loss.item()
        all_acc += accuracy(data.y, output)
        i += 1.0
    
    return all_loss/i, all_acc/i

best_val_loss = np.inf
patience = es_patience

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_step()
    val_loss, val_acc = evaluate(val_loader)
    print(f'Epoch: {epoch:02d}, loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, acc: {train_acc:.5f}, val_acc: {val_acc:.5f}')
    wandb.log({ "train_loss": train_loss, "val_loss": val_loss })
    wandb.log({ "train_acc": train_acc, "val_acc": val_acc })
    if val_loss < best_val_loss :
        best_val_loss = val_loss
        patience = es_patience
        print("New best val_loss {:.4f}".format(val_loss))
        torch.save(model.state_dict(), 'model_best.pt')
    else :
        patience -= 1
        if patience == 0:
            print("Early stopping (best val_loss: {})".format(best_val_loss))
            break

def predict(loader):
    model.eval()
    tar = np.empty((0))
    prd = np.empty((0))
    for data in loader :
        data = data.to(device)
        pred = model(data.x, data.edge_index).cpu().detach().numpy()
        target = data.y.cpu().detach().numpy()
        tar = np.append(tar, target)
        prd = np.append(prd, np.array(pred))
    return tar, prd

del train_loader, val_loader, train_dataset, val_dataset

model.load_state_dict(torch.load('model_best.pt'))
model.eval()

# data_test = CreateGraphDataset("../data_test.root:train", num_samples_test, dist=node_distance)
data_test = mcbm_dataset.MyDataset(dataset="test", N = num_samples_test, reload=reload)
test_loader = DataLoader(data_test, batch_size=batch_size)
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data.x, data.edge_index)
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(output.cpu().numpy())

y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
LogWandb(y_true, y_pred)
