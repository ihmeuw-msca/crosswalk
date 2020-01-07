# -*- coding: utf-8 -*-
"""
    test_data
    ~~~~~~~~~

    Test `data` model for `crosswalk` package.
"""
import numpy as np
import pytest
import crosswalk.data as data


# test case settings
num_obs = 5
num_covs = 3
test_log_ratio = np.random.randn(num_obs)
test_log_ratio_se = np.random.rand(num_obs) + 0.01
test_covs = {
    "cov%i" % i: np.random.randn(num_obs)
    for i in range(num_covs)
}
test_alt_defs = np.arange(5)
test_ref_defs = np.arange(5)[::-1]
test_study_id = np.array([2, 1, 2, 1, 3])


@pytest.mark.parametrize("log_ratio", [test_log_ratio])
@pytest.mark.parametrize("log_ratio_se", [test_log_ratio_se])
@pytest.mark.parametrize("alt_defs", [test_alt_defs])
@pytest.mark.parametrize("ref_defs", [test_ref_defs])
@pytest.mark.parametrize("covs", [test_covs])
@pytest.mark.parametrize("study_id", [None, test_study_id])
def test_cwdata_study_id(log_ratio,
                         log_ratio_se,
                         alt_defs,
                         ref_defs,
                         covs,
                         study_id):
    cwdata = data.CWData(log_ratio,
                         log_ratio_se,
                         alt_defs,
                         ref_defs,
                         covs=covs,
                         study_id=study_id)

    if study_id is not None:
        assert cwdata.num_studies == 3
        assert tuple(cwdata.study_sizes) == (2, 2, 1)
        assert tuple(cwdata.unique_study_id) == (1, 2, 3)
        assert tuple(cwdata.study_id) == (1, 1, 2, 2, 3)
    else:
        assert cwdata.num_studies == 5
        assert tuple(cwdata.study_sizes) == tuple([1]*num_obs)
        assert cwdata.unique_study_id is None


@pytest.mark.parametrize("log_ratio", [test_log_ratio])
@pytest.mark.parametrize("log_ratio_se", [test_log_ratio_se])
@pytest.mark.parametrize("alt_defs", [test_alt_defs])
@pytest.mark.parametrize("ref_defs", [test_ref_defs])
@pytest.mark.parametrize("covs", [test_covs])
@pytest.mark.parametrize("study_id", [None, test_study_id])
@pytest.mark.parametrize("add_intercept", [True, False])
def test_cwdata_add_intercept(log_ratio,
                              log_ratio_se,
                              alt_defs,
                              ref_defs,
                              covs,
                              study_id,
                              add_intercept):
    cwdata = data.CWData(log_ratio,
                         log_ratio_se,
                         alt_defs,
                         ref_defs,
                         covs=covs,
                         study_id=study_id,
                         add_intercept=add_intercept)

    if add_intercept:
        assert "intercept" in cwdata.covs
