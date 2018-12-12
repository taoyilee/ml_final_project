import numpy as np
import matplotlib.pyplot as plt
import mltools as ml

X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
X, Y = ml.shuffleData(X, Y)
percentage = 1
ndata = np.int((percentage / 100) * X.shape[0])
print(f"Training with {ndata}({100*ndata/X.shape[0]:.1f}%) points")
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xt, Yt = Xtr[:ndata], Ytr[:ndata]
print(f"Rescale Xt")
XtS, params = ml.rescale(Xt)
print(f"Rescale Xva")
XvS, _ = ml.rescale(Xva, params)
print(f"Transforms XtS")
XtS_xfm = ml.transforms.fpoly(XtS, 2)
XvS_xfm = ml.transforms.fpoly(XvS, 2)
print(XtS_xfm.shape)
print(XvS_xfm.shape)
reg = np.concatenate((np.linspace(1e-4, 1e-1, 10), np.linspace(0.2, 0.5, 5)))
print(f"Regularization coefficients {reg}")
learners = [ml.linearC.linearClassify() for r in reg]

for i, r in enumerate(reg):
    if i > 0:
        learners[i].theta = learners[i].theta.copy()
    print(f"Training {i} - reg = {r}")
    jsur, lr, epoch = learners[i].train(XtS_xfm, Yt, reg=r, initStep=1, stopTol=1e-5, stopIter=300,
                                        rate_decay=0.8)
    plt.subplot(2, 1, 1)
    plt.plot(range(epoch), jsur)
    plt.title(f"Jsur {i} - reg = {r}")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(range(epoch), lr)
    plt.grid()
    plt.title(f"Learning Rate {i} - reg = {r}")
    plt.show()
training_auc = [l.auc(XtS_xfm, Yt) for l in learners]
validating_auc = [l.auc(XvS_xfm, Yva) for l in learners]
print(training_auc)
print(validating_auc)
plt.plot(reg, training_auc, marker="s", label="Training AUC")
plt.plot(reg, validating_auc, marker="x", label="Validating AUC")
plt.xlabel("Regularization Coefficient")
plt.legend()
plt.grid()
plt.xlim([min(reg), max(reg)])
plt.ylabel("AUROC")
plt.show()
