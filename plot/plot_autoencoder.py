from preprocessing.dataset import SVNHDataset
import numpy as np
from keras.models import model_from_json
from keras.models import Model
import matplotlib.pyplot as plt
import os
import configparser as cp


def plot_ae(config: cp.ConfigParser, tag=None):
    ae_model = config["general"].get("ae_model")
    color_mode = config["general"].get("color_mode")
    tag = config["plot"].get("tag") if tag is None else tag
    exp_dir = f"experiments/{tag}"
    print(f"loading experiment results from {exp_dir}")

    train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
    if color_mode == "grayscale":
        train_set.set_gray_scale()

    with open(os.path.join(exp_dir, f"autoencoder.json"), "r") as f:
        autoencoder = model_from_json(f.read())  # type: Model
    autoencoder.load_weights(os.path.join(exp_dir, f"autoencoder_final.h5"))

    n = 10
    if ae_model == "cnn":
        decoded = autoencoder.predict(train_set.images[:n] / 255)
    else:
        decoded = autoencoder.predict(train_set.images_flatten[:n] / 255)
    plt.figure(figsize=(20, 4))
    plt.gray()
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(train_set.images[i].squeeze() / 255)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        decoded_img = decoded[i]

        if ae_model == "mlp":
            decoded_img = decoded_img.reshape(32, 32, 3 if train_set.color_mode == "rgb" else 1)
        decoded_img = decoded_img.squeeze()
        plt.imshow(decoded_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(os.path.join(exp_dir, "ae_compare.png"))


if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini")
    tag = config["plot"].get("tag")
    plot_ae(config, tag=tag)
