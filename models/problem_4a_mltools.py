import numpy as np
import mltools as ml

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 10

    max_depth = list(range(1, 15))
    learners = [ml.dtree.treeClassify() for _ in max_depth]
    for i, depth in enumerate(max_depth):
        learners[i].train(XtS, Yt, maxDepth=depth, minParent=2, minLeaf=1)

    training_auc = [l.auc(XtS, Yt) for l in learners]
    validating_auc = [l.auc(XvS, Yva) for l in learners]
