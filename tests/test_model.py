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

    alt_dorms = np.random.choice(4, num_obs)
    ref_dorms = 3 - alt_dorms

    study_id = np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

    return crosswalk.data.CWData(obs, obs_se, alt_dorms, ref_dorms,
                                 covs=covs,
                                 study_id=study_id)

@pytest.mark.parametrize('cov_names', [['intercept'],
                                       ['intercept', 'cov0']])
@pytest.mark.parametrize('obs_type', ['diff_log', 'diff_logit'])
def test_cwmodel(cwdata, obs_type, cov_names):
    cwmodel = model.CWModel(cwdata, obs_type, cov_names)
    cwmodel.check()


@pytest.mark.parametrize('cov_names', [['intercept'],
                                       ['intercept', 'cov0']])
@pytest.mark.parametrize('obs_type', ['diff_log', 'diff_logit'])
def test_cwmodel_design_mat(cwdata, obs_type, cov_names):
    cwmodel = model.CWModel(cwdata, obs_type, cov_names)
    cov_mat = np.hstack([cwdata.covs[cov_name][:, None]
                         for cov_name in cwmodel.cov_names])
    design_mat = cwmodel.design_mat
    assert (cwmodel.relation_mat.sum(axis=1) == 0.0).all()
    assert (design_mat.sum(axis=1) == 0.0).all()


@pytest.mark.parametrize('cov_names', [['intercept'],
                                       ['intercept', 'cov0']])
@pytest.mark.parametrize('obs_type', ['diff_log', 'diff_logit'])
@pytest.mark.parametrize('order_prior', [{'intercept':[[1, 2],
                                                       [2, 3]]}])
def test_cwmodel_order_prior(cwdata, obs_type, cov_names, order_prior):
    cwmodel = model.CWModel(cwdata, obs_type, cov_names,
                            order_prior=order_prior)

    constraints_mat = cwmodel.constraint_mat
    assert isinstance(constraints_mat, np.ndarray)
    assert constraints_mat.shape == (2, cwmodel.num_var)
    assert (constraints_mat.sum(axis=1) == 0.0).all()


@pytest.mark.parametrize('cov_names', [['intercept']])
@pytest.mark.parametrize('obs_type', ['diff_log', 'diff_logit'])
def test_cwmodel_predict_alt_vals(cwdata, obs_type, cov_names):
    cwmodel = model.CWModel(cwdata, obs_type, cov_names)
    cwmodel.beta = {
        cwdata.unique_dorms[i]: np.ones(cwmodel.num_var_per_def)
        for i in range(cwdata.num_dorms)
    }
    cwmodel.beta[cwmodel.gold_def] = np.zeros(cwmodel.num_var_per_def)

    ref_vals = 0.5*np.ones(5)
    alt_vals = cwmodel.predict_alt_vals(np.repeat(3 - cwmodel.gold_def, 5),
                                        np.repeat(cwmodel.gold_def, 5),
                                        ref_vals,
                                        add_intercept=True)
    if obs_type == 'diff_log':
        assert np.linalg.norm(alt_vals - np.exp(1.0 + np.log(0.5))) < 1e-8
    elif obs_type == 'diff_logit':
        assert np.linalg.norm(alt_vals - 1.0/(1.0 + np.exp(-1.0))) < 1e-8
    else:
        assert alt_vals is None

