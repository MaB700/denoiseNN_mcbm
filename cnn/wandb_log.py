import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import wandb

class LogWandb():
    def __init__(self, gt, pred):
        self.gt = gt
        self.pred = pred
        
        self.log_all()

    def log_all(self):
        cut_value = 0.5        
        auc = roc_auc_score(self.gt, self.pred)
        wandb.log({"test_auc": auc})
        
        tn, fp, fn, tp = confusion_matrix(y_true=[1 if a_ > cut_value else 0 for a_ in self.gt], \
                                            y_pred=[1 if a_ > cut_value else 0 for a_ in self.pred]).ravel()

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
