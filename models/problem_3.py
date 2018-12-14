import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
from models.common import sample_and_split


class WeightKNNExp(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return np.exp(-self.alpha * x)


class TrainerKNN(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, learner):
        # print(f"Fitting learner {learner}")
        learner.fit(self.x, self.y)
        return learner


def sample_train_knn(k, raw_data, trn_p, dev_p, rescale=True, alpha=0):
    print(f"Resamping...")
    learners = [KNeighborsClassifier(n_neighbors=ki,
                                     weights=WeightKNNExp(alpha), algorithm='auto', leaf_size=30,
                                     p=2, metric='minkowski') for ki in k]
    if rescale:
        x_trn, x_dev, y_trn, y_dev, scaler = sample_and_split(raw_data, train_percentage=trn_p,
                                                              dev_percentage=dev_p, rescale=rescale)
    else:
        x_trn, x_dev, y_trn, y_dev = sample_and_split(raw_data, train_percentage=trn_p,
                                                      dev_percentage=dev_p, rescale=rescale)
    with Pool(8) as p:
        learners = p.map(TrainerKNN(x_trn, y_trn), learners)

    training_auc = [roc_auc_score(y_trn, l.predict_proba(x_trn)[:, 1]) for l in learners]
    validating_auc = [roc_auc_score(y_dev, l.predict_proba(x_dev)[:, 1]) for l in learners]

    return training_auc, validating_auc


def plot_knn_core(k, training_auc, validating_auc, name="3a"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k, np.mean(training_auc, axis=0), marker="s", label="Training AUC", color="blue")
    plt.fill_between(k, np.min(training_auc, axis=0), np.max(training_auc, axis=0), color="blue", alpha=0.1,
                     label='Training (Max-Min)')
    plt.xlabel(f"K Neighbors")
    plt.title(f"AUROC over K Neighbors - Training")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")

    plt.subplot(1, 2, 2)
    plt.plot(k, np.mean(validating_auc, axis=0), marker="x", label="Validating AUC", color="red")
    plt.fill_between(k, np.min(validating_auc, axis=0), np.max(validating_auc, axis=0), color="red", alpha=0.1,
                     label='Validation (Max-Min)')
    plt.xlabel(f"K Neighbors")
    plt.title(f"AUROC over K Neighbors - Validating")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/knn_classifier_{name}.png")
    plt.close("all")
    for i in range(training_auc.shape[0]):
        plt.figure()
        plt.semilogx(k, training_auc[i], marker="s", label="Training AUC", color="blue")
        plt.semilogx(k, validating_auc[i], marker="x", label="Validating AUC", color="red")
        plt.title(f"Sampling #{i+1} - {name}")
        plt.xlabel(f"K Neighbors")
        plt.legend()
        plt.grid()
        plt.ylabel("AUROC")
    plt.savefig(f"plot/knn_classifier_{name}_{i}.png")
    plt.close("all")


def plot_knn(k, train_pickle, dev_pickle, name="3a"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)
    plot_knn_core(k, training_auc, validating_auc, name=name)


def plot_knn_combined_core(k, train_aucs, dev_aucs, name=("3a", "3b")):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    style = ["-", "--"]
    for i, training_auc in enumerate(train_aucs):
        plt.plot(k, np.mean(training_auc, axis=0), markersize=3, marker="s", linestyle=style[i],
                 label=f"Training - {name[i]}",
                 color="blue")
        plt.fill_between(k, np.min(training_auc, axis=0), np.max(training_auc, axis=0), color="blue", alpha=0.1)
    plt.xlabel(f"K Neighbors")
    plt.title(f"AUROC over K Neighbors - Training")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")

    plt.subplot(1, 2, 2)
    for i, dev_aucs in enumerate(dev_aucs):
        plt.plot(k, np.mean(dev_aucs, axis=0), markersize=3, marker="x", linestyle=style[i],
                 label=f"Validating - {name[i]}",
                 color="red")
        plt.fill_between(k, np.min(dev_aucs, axis=0), np.max(dev_aucs, axis=0), color="red", alpha=0.1)

    plt.xlabel(f"K Neighbors")
    plt.title(f"AUROC over K Neighbors - Validation")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/knn_classifier_3_{name[0]}_{name[1]}.png")
    plt.close("all")


def plot_knn_combined(k, train_pickle, dev_pickle, name=("3a", "3b")):
    plt.figure()
    style = ["-", "--"]
    for i, tp in enumerate(train_pickle):
        with open(tp, "rb") as f:
            training_auc = pickle.load(f)
        plt.plot(k, np.mean(training_auc, axis=0), markersize=3, marker="s", linestyle=style[i],
                 label=f"Training - {name[i]}",
                 color="blue")
        plt.fill_between(k, np.min(training_auc, axis=0), np.max(training_auc, axis=0), color="blue", alpha=0.1)

    for i, dp in enumerate(dev_pickle):
        with open(dp, "rb") as f:
            validating_auc = pickle.load(f)
        plt.plot(k, np.mean(validating_auc, axis=0), markersize=3, marker="x", linestyle=style[i],
                 label=f"Validating - {name[i]}",
                 color="red")
        plt.fill_between(k, np.min(validating_auc, axis=0), np.max(validating_auc, axis=0), color="red", alpha=0.1)

    plt.xlabel(f"K Neighbors")
    plt.title(f"AUROC over K Neighbors - {name}")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/knn_classifier_3_{name[0]}_{name[1]}.png")
    plt.close("all")


def plot_knn_alpha(k, alpha, train_pickle, dev_pickle, name="3c"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)
    print(f"training_auc shape: {training_auc.shape}")
    print(f"validating_auc shape: {validating_auc.shape}")
    plt.figure()
    for i, a in enumerate(alpha):
        plt.plot(k, np.mean(training_auc[:, i, :], axis=0), marker="s", label=f"T {a}")
        plt.plot(k, np.mean(validating_auc[:, i, :], axis=0), marker="x", label=f"V {a}")
        plt.fill_between(k, np.min(training_auc[:, i, :], axis=0), np.max(training_auc[:, i, :], axis=0), color="blue",
                         alpha=0.1)
        plt.fill_between(k, np.min(validating_auc[:, i, :], axis=0), np.max(validating_auc[:, i, :], axis=0),
                         color="red", alpha=0.1)
    plt.xlabel(f"K Neighbors")
    plt.title(f"AUROC over K Neighbors - {name}")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/linear_classifier_{name}.png")
    plt.close("all")
    for i in range(training_auc.shape[0]):
        plt.figure()
        for j, a in enumerate(alpha):
            plt.semilogx(k, training_auc[i, j], marker="s", label=f"T {a}")
            plt.semilogx(k, validating_auc[i, j], marker="x", label=f"V {a}")
        plt.title(f"Sampling #{i+1} - {name}")
        plt.xlabel(f"K Neighbors")
        plt.legend()
        plt.grid()
        plt.ylabel("AUROC")
        plt.savefig(f"plot/knn_classifier_{name}_{i}.png")
        plt.close("all")


def plot_knn_alpha_mat_core(k, alpha, training_auc, validating_auc, name="3c"):
    training_auc = np.mean(training_auc, axis=0)
    validating_auc = np.mean(validating_auc, axis=0)
    print(f"training_auc shape: {training_auc.shape}")
    print(f"validating_auc shape: {validating_auc.shape}")
    for n, d in zip(["Training", "Validating"], [training_auc, validating_auc]):
        plt.figure(figsize=(12, 12))
        plt.matshow(d, interpolation="nearest", cmap=plt.get_cmap("jet"))
        ax = plt.gca()
        ax.set_xticks(range(0, len(k), 2))
        ax.set_yticks(range(0, len(alpha), 2))
        ax.set_xticklabels([f"{ki}" for ki in k[0::2]])
        ax.set_yticklabels([f"{ai:.2f}" for ai in alpha[0::2]])
        ax.set_xlabel(f"K Neighbors")
        ax.set_ylabel(f"alpha")
        plt.colorbar()
        ax.set_title(f"KNN {n} - {name}")
        plt.savefig(f"plot/knn_classifier_mat_{n}_{name}.png", dpi=300)
        plt.close("all")


def plot_knn_alpha_mat(k, alpha, train_pickle, dev_pickle, name="3c"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)

    training_auc = np.mean(training_auc, axis=0)
    validating_auc = np.mean(validating_auc, axis=0)
    print(f"training_auc shape: {training_auc.shape}")
    print(f"validating_auc shape: {validating_auc.shape}")
    for n, d in zip(["Training", "Validating"], [training_auc, validating_auc]):
        plt.figure(figsize=(12, 12))
        plt.matshow(d, interpolation="nearest", cmap=plt.get_cmap("jet"))
        ax = plt.gca()
        ax.set_xticks(range(0, len(k), 2))
        ax.set_yticks(range(0, len(alpha), 2))
        ax.set_xticklabels([f"{ki}" for ki in k[0::2]])
        ax.set_yticklabels([f"{ai:.2f}" for ai in alpha[0::2]])
        ax.set_xlabel(f"K Neighbors")
        ax.set_ylabel(f"alpha")
        plt.colorbar()
        ax.set_title(f"KNN {n} - {name}")
        plt.savefig(f"plot/knn_classifier_mat_{n}_{name}.png", dpi=300)
        plt.close("all")
