import scipy.io as sio
import os
import numpy as np
from preprocessing.image import SVHNImage
import PIL.Image
from keras.utils import Sequence


class SVHNSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class SVNHDataset:
    _images = []  # type: np.ndarray
    _greyscale_images = None
    labels = []  # type: np.ndarray
    color_mode = "rgb"

    def __len__(self):
        return len(self.labels)

    def __init__(self, name):
        self.name = name

    def generator(self, batch_size=16, shuffle=False, ae=True, flatten=True):
        idx = range(len(self))
        if shuffle:
            idx = np.random.permutation(range(len(self)))
        x_set = self.images[idx]
        y_set = self.labels[idx]
        if flatten:
            x_set = x_set.reshape(len(self), np.prod(x_set.shape[1:])) / 255.
        if ae:
            return SVHNSequence(x_set, x_set, batch_size=batch_size)
        else:
            return SVHNSequence(x_set, y_set, batch_size=batch_size)

    @property
    def greyscale(self):
        if self._greyscale_images is None:
            raw_grey_scale = list(map(lambda x: np.array(PIL.Image.fromarray(x).convert(mode="L")),
                                      [self._images[i, :, :, :] for i in range(len(self))]))
            self._greyscale_images = np.array(raw_grey_scale)

            # self._greyscale_images = np.moveaxis(self._greyscale_images, 0, -1)[:, :, np.newaxis, :]
        return self._greyscale_images

    @property
    def images(self):
        if self.color_mode == "rgb":
            return self._images
        else:
            return self.greyscale

    @classmethod
    def from_mat(cls, mat_file):
        dict_representation = sio.loadmat(mat_file)
        _, dataset_name = os.path.split(mat_file)
        dataset_name, _ = os.path.splitext(dataset_name)
        dataset = cls(dataset_name)
        dataset._images = np.moveaxis(dict_representation["X"], -1, 0)
        dataset.labels = dict_representation["y"]
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
        if n is None:
            n = len(self.labels)
        for i in range(n):
            img = SVHNImage.from_array(self._images[:, :, :, i], image_id=i, color_mode=self.color_mode)
            img.save(out_dir)

    def __repr__(self):
        return f"{self.name}: {self.images.shape[3]} images in {self.color_mode} mode"
