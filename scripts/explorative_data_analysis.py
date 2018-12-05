from preprocessing.dataset import SVHNDataset
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_set = SVHNDataset.from_mat("dataset/train_32x32.mat")
    print(train_set)
    n = int(0.1 * len(train_set))
    shuffle_idx = np.random.permutation(range(len(train_set)))
    print(f"Label Data Shape: {train_set.labels.shape}")
    print(f"Maximum Value of Label Data: {np.max(train_set.labels[:, :])}")
    print(f"Minimum Value of Label Data: {np.min(train_set.labels[:, :])}")
    print(f"Mean Value of Label Data: {np.mean(train_set.labels[:, :]):.3f}")

    print(f"Image Data Shape: {train_set.images.shape}")
    print(f"Maximum Value of Image Data: {np.max(train_set.images[:, :, :, :])}")
    print(f"Minimum Value of Image Data: {np.min(train_set.images[:, :, :, :])}")
    print(f"Mean Value of Image Data: {np.mean(train_set.images[:, :, :, :]):.3f}")
    sns.distplot(train_set.images[:, :, :, shuffle_idx[:n]].flatten())
    plt.grid()
    plt.savefig(f"images/distplot_{train_set.name}_{train_set.color_mode}.png")
    plt.xlabel("Pixel Value")
    plt.ylabel("Ratio of observations")
    plt.close()

    train_set.set_gray_scale()
    print(f"Image Data Shape: {train_set.images.shape}")
    print(f"Maximum Value of Image Data: {np.max(train_set.images[:, :, :, :])}")
    print(f"Minimum Value of Image Data: {np.min(train_set.images[:, :, :, :])}")
    print(f"Mean Value of Image Data: {np.mean(train_set.images[:, :, :, :]):.3f}")

    print(train_set)
    print(f"Viewing {n} data points")
    # train_set.save_for_viewing("output", n=n)
    # train_set.save_matrix("output", row=10, col=10)

    # # for ni in range(n):
    # #     print(f"Labels #{ni}: {train_set.labels[ni]}")
    #
    dataframe = pd.DataFrame(columns=["image", "label"])
    dataframe["label"] = train_set.labels.flatten()
    dataframe["image"] = train_set.images[0, 0, 0, :]
    dataframe.to_excel(f"{train_set.name}.xlsx")
    sns.countplot(dataframe["label"])
    plt.grid()
    plt.savefig("images/countplot_label.png")

    sns.distplot(train_set.images[:, :, :, shuffle_idx[:n]].flatten())
    plt.grid()
    plt.xlabel("Pixel Value")
    plt.ylabel("Ratio of observations")
    plt.savefig(f"images/distplot_{train_set.name}_{train_set.color_mode}.png")
    plt.close()
