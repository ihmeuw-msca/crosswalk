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
