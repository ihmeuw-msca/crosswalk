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
from xspline import XSpline


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


@pytest.fixture
def dorm_models():
    return [model.CovModel('intercept'),
            model.CovModel('cov0')]


@pytest.fixture
def diff_models():
    return [model.CovModel('cov1'),
            model.CovModel('cov2')]


@pytest.mark.parametrize('obs_type', ['diff_log', 'diff_logit'])
def test_input(cwdata, obs_type, dorm_models, diff_models):
    cwmodel = model.CWModel(cwdata, obs_type,
                            dorm_models=dorm_models,
                            diff_models=diff_models)
    cwmodel.check()


def test_design_mat(cwdata, dorm_models, diff_models):
    obs_type = 'diff_log'
    cwmodel = model.CWModel(cwdata, obs_type,
                            dorm_models=dorm_models,
                            diff_models=diff_models)
    design_mat = cwmodel.design_mat
    assert (cwmodel.relation_mat.sum(axis=1) == 0.0).all()
    assert (design_mat.sum(axis=1) == cwmodel.diff_cov_mat.sum(axis=1)).all()


@pytest.mark.parametrize('dorm_order_prior', [[['1', '2'], ['2', '3']]])
def test_dorm_order_prior(cwdata, dorm_models, diff_models, dorm_order_prior):
    obs_type = 'diff_log'
    cwmodel = model.CWModel(cwdata, obs_type,
                            dorm_models=dorm_models,
                            diff_models=diff_models,
                            dorm_order_prior=dorm_order_prior)

    constraints_mat = cwmodel.constraint_mat
    assert isinstance(constraints_mat, np.ndarray)
    assert constraints_mat.shape == (4, cwmodel.num_vars)
    assert (constraints_mat.sum(axis=1) == 0.0).all()


def test_predict_alt_vals(cwdata, dorm_models, diff_models):
    obs_type = 'diff_log'
    cwmodel = model.CWModel(cwdata, obs_type,
                            dorm_models=dorm_models,
                            diff_models=diff_models)

    cwmodel.fit()
    ref_vals = np.ones(cwdata.num_obs)
    alt_vals = cwmodel.predict_alt_vals(ref_vals)

    true_alt_vals = np.exp(cwmodel.design_mat.dot(cwmodel.beta))

    assert np.allclose(alt_vals, true_alt_vals)

@pytest.mark.parametrize('cov_name', ['cov0', 'cov1'])
@pytest.mark.parametrize('spline', [None,
                                    XSpline(np.linspace(-2.0, 2.0, 3), 3)])
@pytest.mark.parametrize('soln_name', [None, 'cov'])
def test_cov_model(cwdata, cov_name, spline, soln_name):
    cov_model = model.CovModel(cov_name, spline=spline, soln_name=soln_name)

    if spline is None:
        assert cov_model.num_vars == 1
        assert (cov_model.create_design_mat(cwdata) ==
                cwdata.covs[cov_name][:, None]).all()
    else:
        assert cov_model.num_vars == spline.num_spline_bases - 1
        assert (cov_model.create_design_mat(cwdata) ==
                spline.design_mat(cwdata.covs[cov_name])[:, 1:]).all()

    if soln_name is None:
        assert cov_model.soln_name == cov_name
    else:
        assert cov_model.soln_name == soln_name


@pytest.mark.parametrize('cov_name', ['cov0', 'cov1'])
@pytest.mark.parametrize('spline', [None,
                                    XSpline(np.linspace(-2.0, 2.0, 3), 3)])
@pytest.mark.parametrize('soln_name', [None, 'cov'])
@pytest.mark.parametrize('spline_monotonicity', ['increasing',
                                                 'decreasing',
                                                 None])
@pytest.mark.parametrize('spline_convexity', ['convex', 'concave', None])
def test_spline_prior(cwdata, cov_name, spline, soln_name,
                      spline_monotonicity, spline_convexity):
    cov_model = model.CovModel(cov_name,
                               spline=spline,
                               spline_monotonicity=spline_monotonicity,
                               spline_convexity=spline_convexity,
                               soln_name=soln_name)

    constraints_mat = cov_model.create_constraints_mat()

    if not cov_model.use_constraints:
        assert constraints_mat.size == 0
    else:
        num_constraints = np.sum([spline_monotonicity is not None,
                                  spline_convexity is not None])*20
        assert constraints_mat.shape == (num_constraints, cov_model.num_vars)
