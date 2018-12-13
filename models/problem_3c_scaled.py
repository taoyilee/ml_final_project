import numpy as np
from models.problem_3 import sample_train_knn
import pandas as pd
import os

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 3
    k = list(range(1, 51, 2))
    print(k)
    alpha = np.arange(0, 10, 0.5)
    print(alpha)

    csv_out = f"problem_3.csv"
    if not os.path.isfile(csv_out):
        df = pd.DataFrame(columns=["trn_p", "dev_p", "k", "trn_auc", "dev_auc", "alpha", "scaled"])
    else:
        df = pd.read_csv(csv_out)

    for i in range(resamping):
        for j, alpha_i in enumerate(alpha):
            training_auc, validating_auc = sample_train_knn(k, raw_data, trn_p, dev_p, rescale=True, alpha=alpha_i)
            data_table = np.vstack((trn_p * np.ones_like(training_auc), dev_p * np.ones_like(training_auc),
                                    k, training_auc, validating_auc, alpha_i * np.ones_like(training_auc),
                                    True * np.ones_like(training_auc))).T
            df = df.append(
                pd.DataFrame(data_table, columns=["trn_p", "dev_p", "k", "trn_auc", "dev_auc", "alpha", "scaled"]),
                sort=False)

    df.to_csv(csv_out, mode='w', columns=["trn_p", "dev_p", "k", "trn_auc", "dev_auc", "alpha", "scaled"], index=False)
