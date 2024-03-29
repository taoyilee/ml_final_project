import numpy as np
import pickle
import mltools as ml
from homework4.problem_2 import sample_train

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    x_xfm = ml.transforms.fpoly(X, 2)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((x_xfm, Y), axis=1)
    print(f"raw_data has shape of {raw_data.shape}")
    trn_p = 20
    dev_p = 5
    resamping = 10
    reg = np.logspace(-4, 0, 20)
    reg = reg.round(6)
    training_auc = np.zeros((resamping, len(reg)), dtype=float)
    validating_auc = np.zeros((resamping, len(reg)), dtype=float)

    for i in range(resamping):
        training_auc[i], validating_auc[i] = sample_train(reg, raw_data, trn_p, dev_p)

    with open(f"learners/training_auc_2b.pickle", "wb") as f:
        pickle.dump(training_auc, f)
    with open(f"learners/validating_auc_2b.pickle", "wb") as f:
        pickle.dump(validating_auc, f)
