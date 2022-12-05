import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import wandb

def get_hit_average():
    @tf.autograph.experimental.do_not_convert
    def hit_average(data, y_pred):
        y_true = data[:,:,:,0]
        nofHits = tf.math.count_nonzero(tf.greater(y_true,0.01), dtype=tf.float32)
        return (K.sum(y_true*y_pred[:,:,:,0])/nofHits)
    return hit_average   

def hit_average(y_true, y_pred):    
    nofHits = tf.math.count_nonzero(tf.greater(y_true,0.01), dtype=tf.float32)
    return (K.sum(y_true[:,:,:,0]*y_pred[:,:,:,0])/nofHits)   

def noise_average(y_true, y_pred, x):
    noise_mask = tf.subtract(tf.cast(tf.greater(x[:,:,:,0], 0.01), tf.float32), y_true[:,:,:,0])
    nofNoise = tf.math.count_nonzero(tf.greater(noise_mask, 0.01), dtype=tf.float32)
    return (K.sum(noise_mask*y_pred[:,:,:,0])/nofNoise)

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