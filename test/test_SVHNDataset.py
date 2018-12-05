from preprocessing.dataset import SVHNDataset
import numpy as np
import pytest


class TestSVHNDataset:
    def test_name(self):
        ds = SVHNDataset("test")
        assert ds.name == "test"

    def test_from_mat(self):
        ds = SVHNDataset.from_mat("dataset/test_32x32.mat")
        assert ds.name == "test_32x32"

    def test_from_npy(self):
        ds = SVHNDataset.from_npy("dataset_split/arrays/training/rgb_all.npy")
        assert ds.name == "rgb_all"
        assert ds.labels.shape == (71791,)
        assert ds.images.shape == (71791, 32, 32, 3)

    @pytest.mark.parametrize("n", list(range(1, 100, 10)))
    def test_generator_size(self, n):
        ds = SVHNDataset("test")
        ds._images = np.random.randint(low=0, high=255, size=(n, 32, 32, 3))
        ds.labels = np.random.randint(low=1, high=10, size=(n, 1))
        assert n == len(ds.generator(batch_size=1))

    @pytest.mark.parametrize("n, bs", [(100, 16), (3000, 12), (4123, 7)])
    def test_generator_batch(self, n, bs):
        ds = SVHNDataset("test")
        ds._images = np.random.randint(low=0, high=255, size=(n, 32, 32, 3))
        ds.labels = np.random.randint(low=1, high=10, size=(n, 1))
        assert np.ceil(n / bs) == len(ds.generator(batch_size=bs))

    @pytest.mark.parametrize("n, bs", [(1, 16), (100, 16), (3000, 12), (4123, 7)])
    def test_generator_datashape(self, n, bs):
        ds = SVHNDataset("test")
        ds._images = np.random.randint(low=0, high=255, size=(n, 32, 32, 3))
        ds.labels = np.random.randint(low=1, high=10, size=(n, 1))
        ds_gen = ds.generator(batch_size=bs, flatten=False, ae=False)
        for i in range(len(ds_gen)):
            if i == len(ds_gen) - 1:
                if n % bs == 0:
                    assert (bs, 32, 32, 3) == ds_gen[i][0].shape
                    assert (bs, 1) == ds_gen[i][1].shape
                else:
                    assert (n % bs, 32, 32, 3) == ds_gen[i][0].shape
                    assert (n % bs, 1) == ds_gen[i][1].shape
            else:
                assert (bs, 32, 32, 3) == ds_gen[i][0].shape
                assert (bs, 1) == ds_gen[i][1].shape
