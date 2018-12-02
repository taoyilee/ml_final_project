from preprocessing.dataset import SVNHDataset
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
    print(train_set)
    n = int(0.1 * len(train_set))
    shuffle_idx = np.random.permutation(range(len(train_set)))

    sns.distplot(train_set.images[:, :, :, shuffle_idx[:n]].flatten(), label=f"{train_set.color_mode}")
    plt.savefig(f"images/distplot_train_{train_set.color_mode}.png")
    train_set.set_gray_scale()
    sns.distplot(train_set.images[:, :, :, shuffle_idx[:n]].flatten(), label=f"{train_set.color_mode}")
    plt.grid()
    plt.xlabel("Pixel Value")
    plt.ylabel("Ratio of observations")
    plt.legend()
    plt.savefig(f"images/distplot_{train_set.name}.png")
    plt.close()
