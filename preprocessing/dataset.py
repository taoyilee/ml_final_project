import scipy.io as sio
import os
import numpy
import PIL.Image


class SVNHDataset:
    images = []  # type: numpy.ndarray
    labels = []  # type: numpy.ndarray
    color_mode = "rgb"

    def __init__(self, name):
        self.name = name

    @classmethod
    def from_mat(cls, mat_file):
        dict_representation = sio.loadmat(mat_file)
        _, dataset_name = os.path.split(mat_file)
        dataset_name, _ = os.path.splitext(dataset_name)
        dataset = cls(dataset_name)
        dataset.images = dict_representation["X"]
        dataset.labels = dict_representation["y"]
        return dataset

    def to_greyscale(self):
        self.color_mode = "grayscale"

    def save_for_viewing(self, out_dir, n=10):
        os.makedirs(out_dir, exist_ok=True)
        for i in range(n):
            img = PIL.Image.fromarray(self.images[:, :, :, i])
            if self.color_mode == "grayscale":
                img = img.convert(mode="L")
            img.save(os.path.join(out_dir, f"img_{self.color_mode}_{i:05d}.png"))

    def __repr__(self):
        return f"{self.name}: {self.images.shape[3]} images in {self.color_mode} mode"
