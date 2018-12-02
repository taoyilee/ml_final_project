from preprocessing.dataset import SVNHDataset
import numpy as np
import pytest


class TestSVHNDataset:
    def test_name(self):
        ds = SVNHDataset("test")
        assert ds.name == "test"

    def test_from_mat(self):
        ds = SVNHDataset.from_mat("dataset/test_32x32.mat")
        assert ds.name == "test_32x32"

    @pytest.mark.parametrize("n", list(range(1, 100, 10)))
    def test_generator_size(self, n):
        ds = SVNHDataset("test")
        ds._images = np.random.randint(low=0, high=255, size=(32, 32, 3, n))
        ds.labels = np.random.randint(low=1, high=10, size=(n, 1))
        ds_gen = ds.generator(batch_size=1)
        i = 0
        for _ in ds_gen:
            i += 1
        assert n + 1 == i

    @pytest.mark.parametrize("n, bs", [(100, 16), (3000, 12), (4123, 7)])
    def test_generator_batch(self, n, bs):
        ds = SVNHDataset("test")
        ds._images = np.random.randint(low=0, high=255, size=(32, 32, 3, n))
        ds.labels = np.random.randint(low=1, high=10, size=(n, 1))
        ds_gen = ds.generator(batch_size=bs)
        i = 0
        for _ in ds_gen:
            i += 1
        assert n // bs + 1 == i

    @pytest.mark.parametrize("n, bs", [(1, 16), (100, 16), (3000, 12), (4123, 7)])
    def test_generator_datashape(self, n, bs):
        ds = SVNHDataset("test")
        ds._images = np.random.randint(low=0, high=255, size=(32, 32, 3, n))
        ds.labels = np.random.randint(low=1, high=10, size=(n, 1))
        ds_gen = ds.generator(batch_size=bs)
        batch_number = n // bs
        i = 0
        for x, y in ds_gen:
            if i == batch_number:
                assert x.shape == (32, 32, 3, n % bs)
                assert y.shape == (n % bs, 1)
            else:
                assert x.shape == (32, 32, 3, bs)
                assert y.shape == (bs, 1)
            i += 1
