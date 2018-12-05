import numpy as np
import pickle
from homework4.problem_4 import sample_train_dt

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 10
    maxDepth = list(range(3, 16))
    print(maxDepth)

    training_auc = np.zeros((resamping, len(maxDepth)), dtype=float)
    validating_auc = np.zeros((resamping, len(maxDepth)), dtype=float)

    for i in range(resamping):
        training_auc[i], validating_auc[i] = sample_train_dt(maxDepth, raw_data, trn_p, dev_p, minParent=2, minLeaf=1)

    with open(f"learners/training_auc_4a.pickle", "wb") as f:
        pickle.dump(training_auc, f)
    with open(f"learners/validating_auc_4a.pickle", "wb") as f:
        pickle.dump(validating_auc, f)
