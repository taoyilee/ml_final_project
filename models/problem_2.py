import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle


def sample_and_split(raw_data, train_percentage=10, dev_percentage=10):
    np.random.shuffle(raw_data)
    n_trn_data = np.int((train_percentage / 100) * raw_data.shape[0])
    n_dev_data = np.int((dev_percentage / 100) * raw_data.shape[0])
    x_trn_raw = raw_data[:n_trn_data, :-1]
    x_dev_raw = raw_data[n_trn_data:n_trn_data + n_dev_data, :-1]
    y_trn = raw_data[:n_trn_data, -1]
    y_dev = raw_data[n_trn_data:n_trn_data + n_dev_data, -1]
    print(f"Training with {x_trn_raw.shape[0]}({100*x_trn_raw.shape[0]/raw_data.shape[0]:.1f}%) points")
    print(f"Validating with {x_dev_raw.shape[0]}({100*x_dev_raw.shape[0]/raw_data.shape[0]:.1f}%) points")
    scaler = StandardScaler()
    scaler.fit(x_trn_raw)
    x_trn_scaled = scaler.transform(x_trn_raw)
    x_dev_scaled = scaler.transform(x_dev_raw)
    return x_trn_scaled, x_dev_scaled, y_trn, y_dev


class Trainer(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, learner):
        # print(f"Fitting learner {learner}")
        learner.fit(self.x, self.y)
        return learner


def sample_train(reg, raw_data, trn_p, dev_p):
    print(f"Resamping...")
    learners = [SGDClassifier(loss="log", penalty="l2", max_iter=500, alpha=r) for r in reg]
    x_trn_scaled, x_dev_scaled, y_trn, y_dev = sample_and_split(raw_data, train_percentage=trn_p,
                                                                dev_percentage=dev_p)
    with Pool(4) as p:
        learners = p.map(Trainer(x_trn_scaled, y_trn), learners)
    training_auc = [roc_auc_score(y_trn, l.predict_proba(x_trn_scaled)[:, 1]) for l in learners]
    validating_auc = [roc_auc_score(y_dev, l.predict_proba(x_dev_scaled)[:, 1]) for l in learners]
    return training_auc, validating_auc


def plot_linear(train_pickle, dev_pickle, name="2a"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)
    reg = np.logspace(-4, 0, 20)
    reg = reg.round(6)
    plt.figure()
    plt.semilogx(reg, np.mean(training_auc, axis=0), marker="s", label="Training AUC", color="blue")
    plt.semilogx(reg, np.mean(validating_auc, axis=0), marker="x", label="Validating AUC", color="red")
    plt.fill_between(reg, np.min(training_auc, axis=0), np.max(training_auc, axis=0), color="blue", alpha=0.1,
                     label='Training (Max-Min)')
    plt.fill_between(reg, np.min(validating_auc, axis=0), np.max(validating_auc, axis=0), color="red", alpha=0.1,
                     label='Validation (Max-Min)')
    # plt.fill_between(reg, np.mean(training_auc, axis=0) - np.std(training_auc, axis=0),
    #                  np.mean(training_auc, axis=0) + np.std(training_auc, axis=0), color="blue", alpha=0.5,
    #                  label='Training (1sigma)')
    # plt.fill_between(reg, np.mean(validating_auc, axis=0) - np.std(validating_auc, axis=0),
    #                  np.mean(validating_auc, axis=0) + np.std(validating_auc, axis=0), color="red", alpha=0.5,
    #                  label='Validation (1sigma)')
    plt.xlabel(f"L2 Regularization Coefficient")
    plt.title(f"AUROC over L2 Regularization Coefficient - {name}")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/linear_classifier_{name}.png")
    plt.close("all")
    for i in range(training_auc.shape[0]):
        plt.figure()
        plt.semilogx(reg, training_auc[i], marker="s", label="Training AUC", color="blue")
        plt.semilogx(reg, validating_auc[i], marker="x", label="Validating AUC", color="red")
        plt.title(f"Sampling #{i+1} - {name}")
        plt.xlabel(f"L2 Regularization Coefficient")
        plt.legend()
        plt.grid()
        plt.ylabel("AUROC")
        plt.savefig(f"plot/linear_classifier_{name}_{i}.png")
        plt.close("all")
