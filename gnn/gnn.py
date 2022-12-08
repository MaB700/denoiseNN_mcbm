# %%
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from helpers import *
import wandb
wandb.init(entity="mabeyer", project="mrich_denoise") # , mode='disabled'
# %%
batch_size = 512
epochs = 500
es_patience = 5

data = CreateGraphDataset("../TMVA_mcbm/data.root:train", 0)
np.random.seed(123)
idxs = np.random.permutation(len(data))
idx_train, idx_val, idx_test = np.split(idxs, [int(0.6 * len(data)), int(0.99 * len(data))])

train_loader = DataLoader([data[index] for index in idx_train], batch_size=batch_size, shuffle=True)
val_loader = DataLoader([data[index] for index in idx_val], batch_size=batch_size)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(data[0], 32).to(device) # .float() # pass data[0] to get node/edge_feature amount
model = model.to(torch.float)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# %%
def train_step():
    model.train()
    all_loss = 0
    i = 0.0
    for data in train_loader:
        data = data.to(device)
        output = model(data)
        loss = F.binary_cross_entropy(output, data.y, reduction="mean")
        all_loss += loss.item()
        i += 1.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return all_loss/i

def evaluate(loader):
    model.eval()
    all_loss = 0
    i = 0.0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = F.binary_cross_entropy(output, data.y, reduction="mean")
        all_loss += loss.item()
        i += 1.0
    
    return all_loss/i

best_val_loss = np.inf
patience = es_patience

for epoch in range(1, epochs + 1):
    train_loss = train_step()
    val_loss = evaluate(val_loader)
    print(f'Epoch: {epoch:02d}, loss: {train_loss:.5f}, val_loss: {val_loss:.5f}')
    wandb.log({ "train_loss": train_loss, "val_loss": val_loss })

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
        pred = model(data).cpu().detach().numpy()
        target = data.y.cpu().detach().numpy()
        tar = np.append(tar, target)
        prd = np.append(prd, np.array(pred))
    return tar, prd
datax = data[0]
del data, train_loader, val_loader

model.load_state_dict(torch.load('model_best.pt'))
model.eval()

test_data = CreateGraphDataset("../TMVA_mcbm/data_test.root:train", 0)
test_loader = DataLoader(test_data, batch_size=batch_size)
test_gt, test_pred = predict(test_loader)
LogWandb(test_gt, test_pred)

def meanSigma(data):
    n = len(data)
    mean = sum(data)/n
    dev = [(x - mean)**2 for x in data]
    sigma = math.sqrt(sum(dev)/n)
    return mean*1e3, sigma*1e3

test_loader_single = DataLoader(test_data[0:1000], batch_size=1)

def predict_timed(loader, device_timed):
    model.eval()
    times = []
    for data in loader :
        data = data.to(device_timed)
        start = time.process_time()
        pred = model(data)
        stop = time.process_time() - start
        times.append(stop)
    
    return times

model = Net(datax, 32).to(torch.device('cpu')) # .float() # pass data[0] to get node/edge_feature amount
model = model.to(torch.float)
model.load_state_dict(torch.load('model_best.pt'))
model.eval()

t_cpu = predict_timed(test_loader_single, torch.device('cpu'))
mean_cpu, sigma_cpu = meanSigma(t_cpu)
wandb.log({"cpu_time_xmean": mean_cpu})
wandb.log({"cpu_time_xsigma": sigma_cpu})

# t_cuda = predict_timed(test_loader_single, torch.device('cuda'))
# mean_cuda, sigma_cuda = meanSigma(t_cuda)
# wandb.log({"gpu_time_xmean": mean_cuda})
# wandb.log({"gpu_time_xsigma": sigma_cuda})

# test_auc = roc_auc_score(val_gt, val_pred)
# print("Test AUC: {:.4f}".format(test_auc))

# cm = confusion_matrix([1 if a_ > 0.5 else 0 for a_ in val_gt], [1 if a_ > 0.5 else 0 for a_ in val_pred], normalize='true')
# print(cm)
