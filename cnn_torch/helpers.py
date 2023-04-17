import numpy as np
import torch
import uproot
from sklearn.metrics import roc_auc_score, confusion_matrix
import wandb

def accuracy(output, target, input):
    mask = (input > 0.01)
    output = output[mask] > 0.5
    target = target[mask] > 0.5
    return (output == target).float().mean()

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

class LogWandb():
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred
        
        self.log_all()

    def log_all(self):
        cut_value = 0.5        
        auc = roc_auc_score(self.gt, self.pred)
        wandb.log({"test_auc": auc})
        
        tn, fp, fn, tp = confusion_matrix(y_true= self.gt > cut_value, \
                                            y_pred= self.pred > cut_value).ravel()

        wandb.log({"test_acc": (tp+tn)/(tn+fp+tp+fn)}) # accuracy
        wandb.log({"test_sens": tp/(tp+fn)}) # sensitifity
        wandb.log({"test_spec": tn/(tn+fp)}) # specificity
        wandb.log({"test_prec": tp/(tp+fp)}) # precision
        
        wandb.log({"cm": wandb.plot.confusion_matrix(   probs=None,
                                            y_true=[1 if a_ > cut_value else 0 for a_ in self.gt],
                                            preds=[1 if a_ > cut_value else 0 for a_ in self.pred],
                                            class_names=["noise hit", "true hit"],
                                            title="CM")})
        
        len_roc = len(self.gt) if len(self.gt) <= 10000 else 10000
        wandb.log({"roc": wandb.plot.roc_curve( self.gt[:len_roc], 
                                    np.concatenate(((1-self.pred[:len_roc]).reshape(-1,1),self.pred[:len_roc].reshape(-1,1)),axis=1),
                                    classes_to_plot=[1],
                                    title="ROC")})
