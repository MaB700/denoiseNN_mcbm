import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from helpers import *
from networks import *

import wandb
wandb.init(entity="mabeyer", project="GNN_denoise") # , mode='disabled'

# Set up some hyperparameters
num_samples = None
batch_size = 256
max_epochs = 150
patience = 5

train_dataset, val_dataset = torch.utils.data.random_split( MyDataset("../data/data.root", num_samples),
                                                            [0.8, 0.2],
                                                            generator=torch.Generator().manual_seed(123))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Stacked().to(device)
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_val_loss = np.inf
patience = patience
early_stop_counter = 0

for epoch in range(max_epochs):
    # Train the model
    print(f"Epoch {epoch+1} of {max_epochs}")
    #print("-" * 10)
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
    train_loss /= len(train_loader.dataset)/batch_size
    train_acc /= len(train_loader.dataset)/batch_size
    print(f"==> train loss: {train_loss:.5f} Accuracy: {train_acc:.4f}")
    wandb.log({ "train_loss": train_loss, "train_acc": train_acc })
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
    val_acc /= len(val_loader.dataset)/batch_size
    print(f"====> val loss: {val_loss:.5f} Accuracy: {val_acc:.4f}")
    wandb.log({ "val_loss": val_loss, "val_acc": val_acc })
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

wandb.save('model_best_cnn.pt')

del train_loader, val_loader, train_dataset, val_dataset

model.load_state_dict(torch.load('model_best_cnn.pt'))
model.eval()

test_dataset = MyDataset("../data/data.root", num_samples)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

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

LogWandb(y_true, y_pred)

# Save the model to onnx file format
# use cpu here
device = torch.device('cpu')
model = model.to(device)
x = torch.rand(1, 1, 72, 32)
dynamic_axes = {"input": {0: 'batch_size'}, "output": {0: 'batch_size'}}
torch.onnx.export(model, x, "unet_full_mse.onnx", input_names=["input"], output_names=["output"], dynamic_axes=dynamic_axes)
#TODO: save onnx model to wandb


