import numpy as np
import mltools as ml

if __name__ == "__main__":
    # Data Loading
    X = np.genfromtxt('data/X_train.txt', delimiter=None)
    Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
    X, Y = ml.shuffleData(X, Y)

    print(f"Shape of X is {X.shape}")
    print(f"Shape of Y is {Y.shape}")
    for feature in range(X.shape[1]):
        print(f"Min/Max/Mean/Var of feature {feature+1:2d} are:", end="")
        print(f" {np.min(X[:,feature]):+07.3e}", end="/")
        print(f"{np.max(X[:,feature]):+07.3e}", end="/")
        print(f"{np.mean(X[:,feature]):+07.3e}", end="/")
        print(f"{np.var(X[:,feature]):+07.3e}")

    percentage = 20
    ndata = np.int((percentage / 100) * X.shape[0])
    print(f"Training with {ndata}({100*ndata/X.shape[0]:.1f}%) points")

    Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
    Xt, Yt = Xtr[:ndata], Ytr[:ndata]
    XtS, params = ml.rescale(Xt)
    XvS, _ = ml.rescale(Xva, params)  # Normalize the features
    print(f"Shape of XtS is {XtS.shape}")
    print(f"Shape of XvS is {XvS.shape}")
    for feature in range(XtS.shape[1]):
        print(f"Min/Max/Mean/Var of feature {feature+1:2d} in XtS is:", end="")
        print(f" {np.min(XtS[:,feature]):+07.3e}", end="/")
        print(f"{np.max(XtS[:,feature]):+07.3e}", end="/")
        print(f"{np.mean(XtS[:,feature]):+07.3e}", end="/")
        print(f"{np.var(XtS[:,feature]):+07.3e}")

        print(" " * 34 + f"XvS is:", end="")
        print(f" {np.min(XvS[:,feature]):+07.3e}", end="/")
        print(f"{np.max(XvS[:,feature]):+07.3e}", end="/")
        print(f"{np.mean(XvS[:,feature]):+07.3e}", end="/")
        print(f"{np.var(XvS[:,feature]):+07.3e}")
