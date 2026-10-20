# -*- coding: utf-8 -*-
"""
test_data
~~~~~~~~~

Test `data` model for `crosswalk` package.
"""

import crosswalk.data as data
import numpy as np
import pandas as pd
import pytest

# test case settings
num_obs = 5
num_covs = 3


@pytest.fixture()
def df():
    df = pd.DataFrame(
        {
            "obs": np.random.randn(num_obs),
            "obs_se": np.random.rand(num_obs) + 0.01,
            "alt_dorms": np.arange(num_obs),
            "ref_dorms": np.arange(num_obs)[::-1],
            "study_id": np.array([2, 1, 2, 1, 3]),
            "data_id": np.array(["A", "B", "C", "D", "E"]),
        }
    )
    for i in range(num_covs):
        df[f"cov{i}"] = np.random.randn(num_obs)
    return df


@pytest.mark.parametrize("study_id", [None, "study_id"])
def test_cwdata_study_id(df, study_id):
    cwdata = data.CWData(
        df,
        "obs",
        "obs_se",
        "alt_dorms",
        "ref_dorms",
        covs=[f"cov{i}" for i in range(num_covs)],
        study_id=study_id,
    )

    if study_id is not None:
        assert cwdata.num_studies == 3
        assert tuple(cwdata.study_sizes) == (2, 2, 1)
        assert tuple(cwdata.unique_study_id) == (1, 2, 3)
        assert tuple(cwdata.study_id) == (1, 1, 2, 2, 3)
    else:
        assert cwdata.num_studies == 0
        assert cwdata.study_sizes.size == 0
        assert cwdata.unique_study_id is None


@pytest.mark.parametrize("study_id", [None, "study_id"])
@pytest.mark.parametrize("add_intercept", [True, False])
def test_cwdata_add_intercept(df, study_id, add_intercept):
    cwdata = data.CWData(
        df,
        "obs",
        "obs_se",
        "alt_dorms",
        "ref_dorms",
        covs=[f"cov{i}" for i in range(num_covs)],
        study_id=study_id,
        add_intercept=add_intercept,
    )

    if add_intercept:
        assert "intercept" in cwdata.covs.columns


@pytest.mark.parametrize("data_id", [None, "data_id"])
def test_data_id(df, data_id):
    cwdata = data.CWData(
        df,
        "obs",
        "obs_se",
        "alt_dorms",
        "ref_dorms",
        covs=[f"cov{i}" for i in range(num_covs)],
        data_id=data_id,
    )

    if data_id is None:
        assert (np.sort(cwdata.data_id) == np.arange(cwdata.num_obs)).all()
    else:
        assert (np.sort(cwdata.data_id) == np.sort(df[data_id].to_numpy())).all()


@pytest.mark.parametrize(
    "index",
    [
        [0, 1, 2, 3, 4],  # clean index, rows not sorted by study_id
        [3, 0, 4, 1, 2],  # permuted integer labels
        [0, 2, 5, 7, 9],  # gaps, e.g. after filtering
        ["a", "b", "c", "d", "e"],  # string labels, e.g. R row names
    ],
)
def test_sort_by_study_id_keeps_covs_aligned(df, index):
    df = df.copy()
    df["cov0"] = 10.0 * df["obs"]
    df.index = index
    cwdata = data.CWData(
        df,
        "obs",
        "obs_se",
        "alt_dorms",
        "ref_dorms",
        covs=[f"cov{i}" for i in range(num_covs)],
        study_id="study_id",
    )

    assert np.all(np.diff(cwdata.study_id) >= 0)
    assert not cwdata.covs.isna().any().any()
    assert np.allclose(cwdata.covs["cov0"].to_numpy(), 10.0 * cwdata.obs)
    assert list(cwdata.covs.index) == list(range(num_obs))
