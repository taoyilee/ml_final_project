from preprocessing.dataset import SVNHDataset
import numpy as np
import configparser as cp
from datetime import datetime as dt
import os
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    batch_size = config["general"].getint("batch_size")
    ae_model = config["general"].get("ae_model")
    color_mode = config["general"].get("color_mode")
    noise_ratio = config["general"].getfloat("noise_ratio")
    train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
    print(train_set.images_flatten.shape)

    pca = PCA(n_components=5)
    image_pc = pca.fit_transform(train_set.images_flatten / 255.)
    print(image_pc.shape)
    df = pd.DataFrame(columns=["pc0", "pc1", "label"])
    df["label"] = train_set.labels.flatten()
    df["pc0"] = image_pc[:, 0]
    df["pc1"] = image_pc[:, 1]

    for i in range(4):
        for j in range(i+1, 5):
            plt.figure(figsize=(8, 8))
            for k in np.unique(train_set.labels):
                plt.scatter(image_pc[train_set.labels.flatten() == k, i], image_pc[train_set.labels.flatten() == k, j],
                            cmap='jet', s=1, label=f"{k}")
            plt.xlabel(f"Principal Component {i}")
            plt.ylabel(f"Principal Component {j}")
            plt.grid()
            plt.legend()
            plt.savefig(f"images/pca_{train_set.name}_{i}_{j}.png")
            plt.close()
