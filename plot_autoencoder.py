from ae.ae_simple import autoencoder_model
from preprocessing.dataset import SVNHDataset
import numpy as np
from keras.models import model_from_json
from keras.models import Model
import matplotlib.pyplot as plt

if __name__ == "__main__":
    batch_size = 32
    train_set = SVNHDataset.from_mat("dataset/train_32x32.mat")
    train_set.set_gray_scale()

    with open(f"weights/autoencoder_{train_set.color_mode}.json", "r") as f:
        autoencoder = model_from_json(f.read())  # type: Model
    autoencoder.load_weights(f"weights/autoencoder_{train_set.color_mode}.h5")

    n = 10
    decoded = autoencoder.predict(train_set.images[:n].reshape(n, np.prod(train_set.images.shape[1:])) / 255)
    plt.figure(figsize=(20, 4))
    plt.gray()
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(train_set.images[i] / 255)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        decoded_img = decoded[i].reshape(train_set.images.shape[1:])
        print(decoded_img.shape)
        print(decoded_img)
        plt.imshow(decoded_img)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
