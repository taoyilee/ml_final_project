import numpy as np
import pickle
from homework4.problem_3 import sample_train_knn
import pandas as pd
import os

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 3
    alpha = 0.5
    csv_out = f"problem_3.csv"
    if not os.path.isfile(csv_out):
        df = pd.DataFrame(columns=["trn_p", "dev_p", "k", "trn_auc", "dev_auc", "alpha", "scaled"])
    else:
        df = pd.read_csv(csv_out)
    k = list(range(1, 21, 2))
    print(k)

    training_auc = np.zeros((resamping, len(k)), dtype=float)
    validating_auc = np.zeros((resamping, len(k)), dtype=float)

    for i in range(resamping):
        training_auc[i], validating_auc[i] = sample_train_knn(k, raw_data, trn_p, dev_p, rescale=True, alpha=alpha)
        data_table = np.vstack((trn_p * np.ones_like(training_auc[i]), dev_p * np.ones_like(training_auc[i]),
                                k, training_auc[i], validating_auc[i], alpha * np.ones_like(training_auc[i]),
                                True * np.ones_like(training_auc[i]))).T
        df = df.append(
            pd.DataFrame(data_table, columns=["trn_p", "dev_p", "k", "trn_auc", "dev_auc", "alpha", "scaled"]),
            sort=False)

    # ave_training_auc = np.mean(training_auc, axis=0)
    # ave_validating_auc = np.mean(validating_auc, axis=0)

    df.to_csv(csv_out, mode='w', columns=["trn_p", "dev_p", "k", "trn_auc", "dev_auc", "alpha", "scaled"], index=False)

    with open(f"learners/training_auc_3a.pickle", "wb") as f:
        pickle.dump(training_auc, f)
    with open(f"learners/validating_auc_3a.pickle", "wb") as f:
        pickle.dump(validating_auc, f)
