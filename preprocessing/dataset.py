import scipy.io as sio
import os
import numpy as np
from preprocessing.image import SVHNImage
import PIL.Image
from keras.utils import Sequence
import pandas as pd


class SVHNAEFileSequence(Sequence):
    def __init__(self, x_set, batch_size, noise=None, flatten=False, color_mode="rgb"):
        self.x = x_set
        self.batch_size = batch_size
        self.noise = noise
        self.color_mode = color_mode
        self.flatten = flatten

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.color_mode == "rgb":
            batch_x = np.array([np.array(PIL.Image.open(x)) for x in batch_x])
        else:
            batch_x = np.array([np.array(PIL.Image.open(x).convert("L")) for x in batch_x])
        batch_x = batch_x.astype("float32")
        batch_x /= 255.0
        if self.flatten:
            batch_x = batch_x.reshape(batch_x.shape[0], np.prod(batch_x.shape[1:]))

        batch_y = batch_x.copy()
        if self.noise is not None:
            batch_x = batch_x + self.noise * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
        return batch_x, batch_y


class SVHNSequence(Sequence):
    def __init__(self, x_set, y_set, batch_size, noise=None):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.noise = noise

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.noise is not None:
            batch_x = batch_x + self.noise * np.random.normal(loc=0.0, scale=1.0, size=batch_x.shape)
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class SVNHDataset:
    _images = []  # type: np.ndarray
    _greyscale_images = None
    labels = []  # type: np.ndarray
    color_mode = "rgb"
    image_files = None

    def __len__(self):
        return len(self.labels)

    def __init__(self, name):
        self.name = name

    def generator(self, batch_size=16, ae=True, flatten=True, noise=0.5):
        if self.image_files is None:
            x_set = self.images / 255.
            x_set_orig = x_set.copy()
            y_set = self.labels
            if flatten:
                x_set = x_set.reshape(len(self), np.prod(x_set.shape[1:]))
            if ae:
                return SVHNSequence(x_set, x_set_orig, batch_size=batch_size, noise=noise)

            else:
                return SVHNSequence(x_set, y_set, batch_size=batch_size, noise=noise)
        else:
            return SVHNAEFileSequence(self.image_files, batch_size=batch_size, noise=noise, flatten=flatten,
                                      color_mode=self.color_mode)

    @property
    def greyscale(self):
        if self._greyscale_images is None:
            raw_grey_scale = list(map(lambda x: np.array(PIL.Image.fromarray(x).convert(mode="L")),
                                      [self._images[i, :, :, :] for i in range(len(self))]))
            self._greyscale_images = np.array(raw_grey_scale)[:, :, :, np.newaxis]
        return self._greyscale_images

    @property
    def images_shape(self):
        if self.image_files is None:
            return self.images.shape
        else:
            if self.color_mode == "rgb":
                return np.array(PIL.Image.open(self.image_files[0])).shape
            else:
                return np.array(PIL.Image.open(self.image_files[0]).convert("L")).shape

    @property
    def images(self):
        if self.color_mode == "rgb":
            return self._images
        else:
            return self.greyscale

    @property
    def images_flatten(self):
        if self.color_mode == "rgb":
            return self._images.reshape(len(self), 32 * 32 * 3)
        else:
            return self.greyscale.reshape(len(self), 32 * 32)

    @classmethod
    def from_mat(cls, mat_file):
        dict_representation = sio.loadmat(mat_file)
        _, dataset_name = os.path.split(mat_file)
        dataset_name, _ = os.path.splitext(dataset_name)
        dataset = cls(dataset_name)
        dataset._images = np.moveaxis(dict_representation["X"], -1, 0)
        dataset.labels = dict_representation["y"]
        return dataset

    @classmethod
    def from_csv(cls, csv_file, image_root_dir=None):
        # dict_representation = sio.loadmat(mat_file)
        _, dataset_name = os.path.split(csv_file)
        dataset_name, _ = os.path.splitext(dataset_name)
        dataset = cls(dataset_name)

        df = pd.read_csv(csv_file)
        # np.array(PIL.Image.fromarray(x).convert(mode="L"))
        dataset.image_files = [os.path.join(image_root_dir, f) for f in df["file_names"]]
        dataset.labels = df["labels"]
        return dataset

    def set_gray_scale(self):
        print(f"Dataset set to greyscale mode")
        self.color_mode = "grayscale"

    def save_matrix(self, out_dir, row=5, col=5):
        out_image = np.zeros((row * 32 + row + 1, col * 32 + col + 1, 3 if self.color_mode == "rgb" else 1),
                             dtype=np.uint8)
        k = 0
        for i in range(row):
            for j in range(row):
                start_x = 1 + i * 32 + i
                start_y = 1 + j * 32 + j
                out_image[start_x:start_x + 32, start_y:start_y + 32, :] = self.images[:, :, :, k]
                k += 1
        PIL.Image.fromarray(out_image.squeeze()).save(os.path.join(out_dir, f"img_matrix_{self.color_mode}.png"))

    def save_for_viewing(self, out_dir, n=None):
        os.makedirs(out_dir, exist_ok=True)
        file_names = []
        if n is None:
            n = len(self.labels)
        for i in range(n):
            img = SVHNImage.from_array(self._images[i, :, :, :], image_id=i, color_mode=self.color_mode)
            file_names.append(img.save(out_dir))
        return file_names

    def __repr__(self):
        return f"{self.name}: {self.images.shape[3]} images in {self.color_mode} mode"
