import numpy as np
import pickle
from homework4.problem_3 import sample_train_knn

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 10
    k = list(range(1, 50, 2))
    print(k)
    alpha = np.linspace(0, 5, len(k))
    alpha = alpha.round(2)
    print(alpha)

    training_auc = np.zeros((resamping, len(alpha), len(k)), dtype=float)
    validating_auc = np.zeros((resamping, len(alpha), len(k)), dtype=float)

    for i in range(resamping):
        for j, alphai in enumerate(alpha):
            training_auc[i, j], validating_auc[i, j] = sample_train_knn(k, raw_data, trn_p, dev_p, rescale=False,
                                                                        alpha=alphai)

    with open(f"learners/training_auc_3c_unscaled.pickle", "wb") as f:
        pickle.dump(training_auc, f)
    with open(f"learners/validating_auc_3c_unscaled.pickle", "wb") as f:
        pickle.dump(validating_auc, f)
