import pytest
import mltools as ml
import numpy as np


class TestFpoly:
    @pytest.mark.parametrize("x,y", [
        (np.array([[1, 2, 3, 4], [4, 5, 6, 7]]).reshape(2, 4),
         [[1, 1, 2, 3, 4, 1, 4, 9, 16, 2, 3, 4, 6, 8, 12],
          [1, 4, 5, 6, 7, 16, 25, 36, 49, 20, 24, 28, 30, 35, 42]])])
    def test_fpoly(self, x, y):
        assertion = np.array_equal(y, ml.transforms.fpoly(x, 2))
        assert assertion


class TestPoly_feature_1d:
    @pytest.mark.parametrize("x,y",
                             [([1, 2, 3], [1, 1, 2, 3, 1, 4, 9, 2, 3, 6]),
                              ([1, 2, 3, 4, 5, 6, 7],
                               [1, 1, 2, 3, 4, 5, 6, 7, 1, 4, 9, 16, 25, 36, 49, 2, 3, 4, 5, 6, 7, 6, 8, 10, 12, 14, 12,
                                15, 18, 21, 20, 24, 28, 30, 35, 42])])
    def test_poly_1d(self, x, y):
        assert np.array_equal(y, ml.transforms.poly_feature_deg2(x))

    def test_poly_1d_badinput(self):
        with pytest.raises(ValueError):
            ml.transforms.poly_feature_deg2([[1, 2, 3, 2, 4, 9], [1, 2, 3, 2, 4, 9]])
