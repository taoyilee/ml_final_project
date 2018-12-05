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


def plot_dnn_mat(dnn_hidden, dnn_nodes, training_auc, validating_auc):
    print(f"training_auc shape: {training_auc.shape}")
    print(f"validating_auc shape: {validating_auc.shape}")
    for n, d in zip(["Training", "Validating"], [training_auc, validating_auc]):
        plt.figure(figsize=(12, 12))
        ml_max_idx, mp_max_idx = np.unravel_index(d.argmax(), d.shape)
        plt.matshow(d, interpolation="nearest", cmap=plt.get_cmap("BuGn"))
        ax = plt.gca()
        ax.add_patch(Rectangle((mp_max_idx - 0.5, ml_max_idx - 0.5), width=1, height=1))
        plt.text(mp_max_idx, ml_max_idx, f"{d[ml_max_idx,mp_max_idx]:.4f}")
        ax.set_xticks(range(0, len(dnn_hidden)))
        ax.set_yticks(range(0, len(dnn_nodes)))
        ax.set_xticklabels([f"{mpi}" for mpi in dnn_hidden])
        ax.set_yticklabels([f"{mli}" for mli in dnn_nodes])
        ax.set_xlabel(f"Hidden layers")
        ax.set_ylabel(f"Nodes Per Hidden Layer")
        plt.colorbar()
        ax.set_title(f"MLP Hyper Parameters - {n}")
        plt.savefig(f"plot/dnn_classifier_5a_{n}.png", dpi=300)
        plt.close("all")
