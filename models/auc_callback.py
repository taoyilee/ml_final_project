import keras.backend as K
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


class auc_callback(Callback):
    def __init__(self, x_trn, y_trn, x_dev=None, y_dev=None):
        super(Callback, self).__init__()
        self.x_dev = x_dev
        self.y_dev = y_dev
        self.x_trn = x_trn
        self.y_trn = y_trn

    def on_train_begin(self, logs={}):
        print(f"** auc_callback is ready")
        logs['last_auc'] = 0.5

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        lr = float(K.eval(self.model.optimizer.lr))
        print(f"current learning rate: {lr:.3e}")
        print(f"*** epoch#{epoch + 1} dev auroc ***")
        y_pred_trn = self.model.predict(self.x_trn)
        trn_auc = roc_auc_score(self.y_trn, y_pred_trn)
        logs['last_trn_auc'] = trn_auc
        logs['lr'] = lr
        if self.x_dev is not None:
            y_pred_dev = self.model.predict(self.x_dev)
            dev_auc = roc_auc_score(self.y_dev, y_pred_dev)
            logs['last_dev_auc'] = dev_auc
            print(f"AUC(trn/dev) = {trn_auc:.4f}/{dev_auc:.4f}")
        else:
            print(f"AUC(trn) = {trn_auc:.4f}")
