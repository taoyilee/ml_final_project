import numpy as np
import mltools as ml

if __name__ == "__main__":
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)[:, np.newaxis]
    raw_data = np.concatenate((X, Y), axis=1)
    trn_p = 20
    dev_p = 5
    resamping = 10

    k = list(range(1, 30))
    learners = [ml.knn.knnClassify() for ki in k]
    for i, ki in enumerate(k):
        learners[i].train(XtS, Yt, K=ki, alpha=0.5)

    training_auc = [l.auc(XtS, Yt) for l in learners]
    validating_auc = [l.auc(XvS, Yva) for l in learners]
