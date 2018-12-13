import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle
from homework4.common import sample_and_split
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


class TrainerDT(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, learner):
        learner.fit(self.x, self.y)
        return learner


def sample_train_dt(max_depth, raw_data, trn_p, dev_p, rescale=True, minParent=2, minLeaf=1):
    print(f"Resamping...")
    learners = [DecisionTreeClassifier(
        criterion='entropy', splitter='best', max_depth=md, min_samples_split=minParent, min_samples_leaf=minLeaf,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False) for md in max_depth]
    x_trn, x_dev, y_trn, y_dev = sample_and_split(raw_data, train_percentage=trn_p,
                                                  dev_percentage=dev_p, rescale=rescale)
    with Pool(4) as p:
        learners = p.map(TrainerDT(x_trn, y_trn), learners)
    training_auc = [roc_auc_score(y_trn, l.predict_proba(x_trn)[:, 1]) for l in learners]
    validating_auc = [roc_auc_score(y_dev, l.predict_proba(x_dev)[:, 1]) for l in learners]
    return training_auc, validating_auc


def sample_train_dt_minParent(minParent, raw_data, trn_p, dev_p, rescale=True, max_depth=15, minLeaf=1):
    print(f"Resamping...")
    learners = [DecisionTreeClassifier(
        criterion='entropy', splitter='best', max_depth=max_depth, min_samples_split=mp, min_samples_leaf=minLeaf,
        min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False) for mp in minParent]
    x_trn, x_dev, y_trn, y_dev = sample_and_split(raw_data, train_percentage=trn_p,
                                                  dev_percentage=dev_p, rescale=rescale)
    with Pool(4) as p:
        learners = p.map(TrainerDT(x_trn, y_trn), learners)
    training_auc = [roc_auc_score(y_trn, l.predict_proba(x_trn)[:, 1]) for l in learners]
    validating_auc = [roc_auc_score(y_dev, l.predict_proba(x_dev)[:, 1]) for l in learners]
    return training_auc, validating_auc


def plot_dt(mdepth, train_pickle, dev_pickle, name="4a"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(mdepth, np.mean(training_auc, axis=0), marker="s", label="Training AUC", color="blue")
    plt.fill_between(mdepth, np.min(training_auc, axis=0), np.max(training_auc, axis=0), color="blue", alpha=0.1,
                     label='Training (Max-Min)')
    plt.xlabel(f"Max Depth")
    plt.title(f"AUROC over Max Depth - Training")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.subplot(1, 2, 2)
    plt.plot(mdepth, np.mean(validating_auc, axis=0), marker="x", label="Validating AUC", color="red")
    plt.fill_between(mdepth, np.min(validating_auc, axis=0), np.max(validating_auc, axis=0), color="red", alpha=0.1,
                     label='Validation (Max-Min)')
    plt.xlabel(f"Max Depth")
    plt.title(f"AUROC over Max Depth - Validation")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/dt_classifier_{name}.png")
    plt.close("all")
    for i in range(training_auc.shape[0]):
        plt.figure()
        plt.semilogx(mdepth, training_auc[i], marker="s", label="Training AUC", color="blue")
        plt.semilogx(mdepth, validating_auc[i], marker="x", label="Validating AUC", color="red")
        plt.title(f"Sampling #{i+1} - {name}")
        plt.xlabel(f"Max Depth")
        plt.legend()
        plt.grid()
        plt.ylabel("AUROC")
        plt.savefig(f"plot/dt_classifier_{name}_{i}.png")
        plt.close("all")


def plot_dt_mLeaf(mdepth, mLeaf, train_pickle, dev_pickle, name="4b_mLeaf"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)
    plt.figure(figsize=(10, 5))
    cmap = cm.get_cmap("copper")
    norm = Normalize(vmin=0, vmax=len(mLeaf))
    for i, ml in enumerate(mLeaf):
        plt.subplot(1, 2, 1)
        plt.plot(mdepth, np.mean(training_auc[:, i], axis=0), marker="s", label=f"{ml}", color=cmap(norm(i)))
        plt.fill_between(mdepth, np.min(training_auc[:, i], axis=0), np.max(training_auc[:, i], axis=0),
                         color=cmap(norm(i)),
                         alpha=0.1)
        plt.subplot(1, 2, 2)
        plt.plot(mdepth, np.mean(validating_auc[:, i], axis=0), marker="x", label=f"{ml}", color=cmap(norm(i)))
        plt.fill_between(mdepth, np.min(validating_auc[:, i], axis=0), np.max(validating_auc[:, i], axis=0),
                         color=cmap(norm(i)), alpha=0.1)
    plt.subplot(1, 2, 1)
    plt.xlabel(f"Max Depth")
    plt.title(f"Training - Sweep minLeaf")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.subplot(1, 2, 2)
    plt.xlabel(f"Max Depth")
    plt.title(f"Validation - Sweep minLeaf")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/dt_classifier_mLeaf_{name}.png")
    plt.close("all")


def plot_dt_mParent(mdepth, mParent, train_pickle, dev_pickle, name="4b_mLeaf"):
    with open(train_pickle, "rb") as f:
        training_auc = pickle.load(f)
    with open(dev_pickle, "rb") as f:
        validating_auc = pickle.load(f)
    plt.figure(figsize=(10, 5))
    cmap = cm.get_cmap("copper")
    norm = Normalize(vmin=0, vmax=len(mParent))
    for i, mp in enumerate(mParent):
        plt.subplot(1, 2, 1)
        plt.plot(mdepth, np.mean(training_auc[:, i], axis=0), marker="s", label=f"{mp}", color=cmap(norm(i)))
        plt.fill_between(mdepth, np.min(training_auc[:, i], axis=0), np.max(training_auc[:, i], axis=0),
                         color=cmap(norm(i)), alpha=0.1)
        plt.subplot(1, 2, 2)
        plt.plot(mdepth, np.mean(validating_auc[:, i], axis=0), marker="x", label=f"{mp}", color=cmap(norm(i)))
        plt.fill_between(mdepth, np.min(validating_auc[:, i], axis=0), np.max(validating_auc[:, i], axis=0),
                         color=cmap(norm(i)), alpha=0.1)
    plt.subplot(1, 2, 1)
    plt.xlabel(f"Max Depth")
    plt.title(f"Training - Sweep minParent")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.subplot(1, 2, 2)
    plt.xlabel(f"Max Depth")
    plt.title(f"Validation - Sweep minParent")
    plt.legend()
    plt.grid()
    plt.ylabel("AUROC")
    plt.savefig(f"plot/dt_classifier_mParent_{name}.png")
    plt.close("all")


def plot_dt_mat(minParent, minLeaf, train_pickle, dev_pickle, name="4c"):
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
        ml_max_idx, mp_max_idx = np.unravel_index(d.argmax(), d.shape)
        print(
            f"minParent={minParent[mp_max_idx]}({mp_max_idx}) mLeaf={minLeaf[ml_max_idx]}(#{ml_max_idx}) max = {d[ml_max_idx,mp_max_idx]:.4f}")
        plt.matshow(d, interpolation="nearest", cmap=plt.get_cmap("gray"))
        ax = plt.gca()
        ax.add_patch(Rectangle((mp_max_idx - 0.5, ml_max_idx - 0.5), width=1, height=1))
        plt.text(mp_max_idx, ml_max_idx, f"{d[ml_max_idx,mp_max_idx]:.4f}")
        ax.set_xticks(range(0, len(minParent), 2))
        ax.set_yticks(range(0, len(minLeaf), 2))
        ax.set_xticklabels([f"{mpi}" for mpi in minParent[0::2]])
        ax.set_yticklabels([f"{mli}" for mli in minLeaf[0::2]])
        ax.set_xlabel(f"minParent")
        ax.set_ylabel(f"minLeaf")
        plt.colorbar()
        ax.set_title(f"KNN {n} - {name}")
        plt.savefig(f"plot/dt_classifier_mat_{n}_{name}.png", dpi=300)
        plt.close("all")
