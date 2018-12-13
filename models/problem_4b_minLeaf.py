import numpy as np
import pickle
from models.problem_4 import sample_train_dt
from models.problem_4 import plot_dt_mLeaf

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 8
    maxDepth = list(range(5, 15))
    print(maxDepth)

    minLeaf = list(range(6, 9))
    print(minLeaf)

    training_auc = np.zeros((resamping, len(minLeaf), len(maxDepth)), dtype=float)
    validating_auc = np.zeros((resamping, len(minLeaf), len(maxDepth)), dtype=float)

    for i in range(resamping):
        for j, ml in enumerate(minLeaf):
            training_auc[i, j], validating_auc[i, j] = sample_train_dt(maxDepth, raw_data, trn_p, dev_p,
                                                                       minParent=2, minLeaf=ml)

    with open(f"learners/training_auc_minLeaf_4b.pickle", "wb") as f:
        pickle.dump(training_auc, f)
    with open(f"learners/validating_auc_minLeaf_4b.pickle", "wb") as f:
        pickle.dump(validating_auc, f)

    plot_dt_mLeaf(maxDepth, minLeaf, "learners/training_auc_minLeaf_4b.pickle",
                  f"learners/validating_auc_minLeaf_4b.pickle",
                  name="4b_minLeaf")
