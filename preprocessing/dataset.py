import scipy.io as sio
import os
import numpy as np
from keras.utils import to_categorical
import PIL.Image
from keras.utils import Sequence
import copy
import matplotlib.pyplot as plt


class SVHNAESequence(Sequence):
    def __init__(self, x_set, batch_size, noise=None):
        self.x = x_set
        self.batch_size = batch_size
        self.noise = noise

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] / 255.0
        batch_y = batch_x.copy()
        if self.noise is not None:
            batch_x = batch_x + self.noise * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
        return batch_x, batch_y


class SVHNSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, noise=None, to_categorical=True):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.noise = noise
        self.to_categorical = to_categorical

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size] / 255.0
        if self.noise is not None:
            batch_x = batch_x + self.noise * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.to_categorical:  # convert to one hot encoding
            batch_y = batch_y - 1
            batch_y = to_categorical(batch_y, num_classes=10)
        return batch_x, batch_y


class SVHNDataset:
    _images = []  # type: np.ndarray
    _greyscale_images = None
    labels = []  # type: np.ndarray
    image_files = None

    def __len__(self):
        return len(self.labels)

    def __init__(self, name, images, labels):
        self.name = name
        self._images = images
        self.labels = labels

    def generator(self, batch_size=16, ae=True, flatten=True, noise=0.001):
        x = self.images_flatten if flatten else self.images
        if ae:
            return SVHNAESequence(x, batch_size=batch_size, noise=noise)
        else:
            y_set = self.labels
            return SVHNSequence(x, y_set, batch_size=batch_size, noise=noise)

    @property
    def images_shape(self):
        return self._images.shape[1:]

    @property
    def images(self):
        return self._images

    @property
    def images_flatten(self):
        return self._images.reshape(len(self), np.prod(self.images_shape))

    @classmethod
    def from_mat(cls, mat_file):
        dict_representation = sio.loadmat(mat_file)
        _, dataset_name = os.path.split(mat_file)
        dataset_name, _ = os.path.splitext(dataset_name)
        images = np.moveaxis(dict_representation["X"], -1, 0)
        labels = dict_representation["y"]
        return cls(dataset_name, images, labels)

    @classmethod
    def from_npy(cls, npy_file):
        dataset_array = np.load(npy_file)
        _, dataset_name = os.path.split(npy_file)
        dataset_name, _ = os.path.splitext(dataset_name)
        images = dataset_array[:, :, :, 0:3]
        labels = dataset_array[:, 0, 0, 3]
        print(f"Loading from {npy_file}")
        print(f"Image has shape {images.shape}")
        print(f"Label has shape {labels.shape}")
        return cls(dataset_name, images, labels)

    @property
    def color_mode(self):
        if self._images.shape[3] == 3:
            return "rgb"
        else:
            return "grayscale"

    def __repr__(self):
        return f"{self.name}: {self.images.shape[3]} images in {self.color_mode} mode"


class ColorConverter:
    MAPPING = {"grayscale": "L"}

    def __init__(self, target_color="grayscale"):
        self.target_color = target_color

    def transform(self, dataset: SVHNDataset):
        print(f"Converting {dataset.name} to {self.target_color}")
        new_dataset = copy.copy(dataset)  # type:SVHNDataset

        def convert_image(x, mode="L"):
            x_conv = np.array(PIL.Image.fromarray(x).convert(mode))
            if mode == "L":
                x_conv = x_conv[..., np.newaxis]
            return x_conv

        image_converted = np.array([convert_image(new_dataset.images[i, ...], self.MAPPING[self.target_color]) for i in
                                    range(len(new_dataset))])
        new_dataset._images = image_converted
        return new_dataset


class SVHNPlotter:
    def __init__(self, output_dir, print_label=True, label_size=12):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.print_label = print_label
        self.label_size = label_size

    def plot_image(self, image: np.ndarray, label):
        plt.imshow(image.squeeze())
        plt.axis('off')
        if image.shape[2] == 1:
            plt.gray()
        if self.print_label:
            plt.text(0, 0, f"{label}", color="red", fontsize=self.label_size)

    def save_images(self, dataset: SVHNDataset, n=None):
        file_names = []
        if n is None:
            n = len(dataset)
        for i in range(n):
            plt.figure()
            self.plot_image(dataset.images[i], dataset.labels[i])
            file_name = os.path.join(self.output_dir, f"image_{dataset.name}_{i:05d}.png")
            print(f"Saving to {file_name}")
            plt.savefig(file_name)
            plt.close()
            file_names.append(file_name)
        return file_names

    def save_mosaic(self, dataset: SVHNDataset, row=10, col=10):
        k = 1
        plt.figure(figsize=(1.2 * row, 1.2 * col))
        for i in range(row):
            for j in range(row):
                plt.subplot(row, col, k)
                self.plot_image(dataset.images[k - 1], dataset.labels[k - 1])
                k += 1
        file_name = os.path.join(self.output_dir, f"mosaic_{dataset.name}.png")
        plt.savefig(file_name)
        plt.close()
        return file_name
