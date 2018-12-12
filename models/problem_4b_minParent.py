import numpy as np
import pickle
from homework4.problem_4 import sample_train_dt, plot_dt_mParent

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 6
    maxDepth = list(range(5, 14))
    print(maxDepth)

    minParent = list(range(6, 13))
    print(minParent)

    training_auc = np.zeros((resamping, len(minParent), len(maxDepth)), dtype=float)
    validating_auc = np.zeros((resamping, len(minParent), len(maxDepth)), dtype=float)

    for i in range(resamping):
        for j, mp in enumerate(minParent):
            training_auc[i, j], validating_auc[i, j] = sample_train_dt(maxDepth, raw_data, trn_p, dev_p,
                                                                       minParent=mp, minLeaf=1)

    with open(f"learners/training_auc_minParent_4b.pickle", "wb") as f:
        pickle.dump(training_auc, f)
    with open(f"learners/validating_auc_minParent_4b.pickle", "wb") as f:
        pickle.dump(validating_auc, f)

    plot_dt_mParent(maxDepth, minParent, "learners/training_auc_minParent_4b.pickle",
                    f"learners/validating_auc_minParent_4b.pickle",
                    name="4b_minParent")
