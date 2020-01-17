# -*- coding: utf-8 -*-
"""
    test_utils
    ~~~~~~~~~~
    Test `utils` module of `crosswalk` package.
"""
import numpy as np
import pytest
import crosswalk.utils as utils


@pytest.mark.parametrize("x",
                         [[1]*3,
                          np.ones(3),
                          np.arange(3),
                          np.zeros(3) + 0j,
                          np.array([np.nan, 0.0, 0.0]),
                          np.array([np.inf, 0.0, 0.0]),
                          np.array(['a', 'b', 'c'])])
@pytest.mark.parametrize("shape", [None, (3,), (4,)])
def test_is_numerical_array(x, shape):
    ok = utils.is_numerical_array(x, shape=shape)
    if (
            not isinstance(x, np.ndarray) or
            not np.issubdtype(x.dtype, np.number) or
            np.isnan(x).any() or
            np.isinf(x).any() or
            (shape is not None and shape != (3,))
    ):
        assert not ok
    else:
        assert ok


@pytest.mark.parametrize("sizes", [np.array([1, 2, 3])])
@pytest.mark.parametrize("indices", [[range(0, 1), range(1, 3), range(3, 6)]])
def test_sizes_to_indices(sizes, indices):
    my_indices = utils.sizes_to_indices(sizes)
    assert all([my_indices[i] == indices[i] for i in range(len(sizes))])


@pytest.mark.parametrize("x", [np.array([1, 1, 2, 2, 3])])
@pytest.mark.parametrize("num_x", [3])
@pytest.mark.parametrize("x_sizes", [np.array([2, 2, 1])])
@pytest.mark.parametrize("unique_x", [np.array([1, 2, 3])])
def test_array_structure(x, num_x, x_sizes, unique_x):
    my_num_x, my_x_sizes, my_unique_x = utils.array_structure(x)

    assert my_num_x == num_x
    assert (my_x_sizes == x_sizes).all()
    assert (my_unique_x == unique_x).all()
