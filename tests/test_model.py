# -*- coding: utf-8 -*-
"""
    test_model
    ~~~~~~~~~~

    Test `model` module for the `crosswalk` package.
"""
import numpy as np
import pytest
import crosswalk
import crosswalk.model as model


@pytest.fixture
def cwdata():
    num_obs = 10
    num_covs = 3

    obs = np.random.randn(num_obs)
    obs_se = 0.1 + np.random.rand(num_obs)*0.1

    covs = {
        'cov%i' % i: np.random.randn(num_obs)
        for i in range(num_covs)
    }

    alt_defs = np.random.choice(4, num_obs)
    ref_defs = 3 - alt_defs

    study_id = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    return crosswalk.data.CWData(obs, obs_se, alt_defs, ref_defs,
                                 covs=covs,
                                 study_id=study_id)

@pytest.mark.parametrize('cov_names', [['intercept'],
                                       ['intercept', 'cov0']])
@pytest.mark.parametrize('obs_type', ['diff_log', 'diff_logit'])
def test_cwmodel(cwdata, obs_type, cov_names):
    cwmodel = model.CWModel(cwdata, obs_type, cov_names)

    assert (np.sum(cwmodel.design_mat, axis=1) == 0).all()
