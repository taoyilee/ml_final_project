import numpy as np
import pickle
from models.problem_4 import sample_train_dt_minParent, plot_dt_mat

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 10
    minParent = list(range(2, 20))
    print(minParent)
    minLeaf = list(range(2, 20))
    print(minLeaf)
    training_auc = np.zeros((resamping, len(minLeaf), len(minParent)), dtype=float)
    validating_auc = np.zeros((resamping, len(minLeaf), len(minParent)), dtype=float)

    for maxDepth in [6, 7, 8]:
        for i in range(resamping):
            for j, ml in enumerate(minLeaf):
                training_auc[i, j], validating_auc[i, j] = sample_train_dt_minParent(minParent, raw_data, trn_p, dev_p,
                                                                                     max_depth=maxDepth, minLeaf=ml)

        with open(f"learners/training_auc_minParent_4c_{maxDepth}.pickle", "wb") as f:
            pickle.dump(training_auc, f)
        with open(f"learners/validating_auc_minParent_4c_{maxDepth}.pickle", "wb") as f:
            pickle.dump(validating_auc, f)

        plot_dt_mat(minParent, minLeaf, f"learners/training_auc_minParent_4c_{maxDepth}.pickle",
                    f"learners/validating_auc_minParent_4c_{maxDepth}.pickle",
                    name=f"4c_mDepth_{maxDepth}")
