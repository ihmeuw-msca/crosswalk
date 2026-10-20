# -*- coding: utf-8 -*-
"""
test_model
~~~~~~~~~~

Test `model` module for the `crosswalk` package.
"""

import warnings

import crosswalk
import crosswalk.model as model
import numpy as np
import pandas as pd
import pytest
from xspline import XSpline


@pytest.fixture
def cwdata():
    np.random.seed(123)
    num_obs = 10
    num_covs = 3
    df = pd.DataFrame(
        {
            "obs": np.random.randn(num_obs),
            "obs_se": 0.1 + np.random.rand(num_obs) * 0.1,
            "alt_dorms": np.random.choice(4, num_obs),
            "study_id": np.array([1, 1, 2, 2, 2, 2, 3, 3, 3, 3]),
        }
    )
    df["ref_dorms"] = np.array(
        [3 if alt_dorm != 3 else 0 for alt_dorm in df["alt_dorms"]]
    )
    for i in range(num_covs):
        df[f"cov{i}"] = np.random.randn(num_obs)

    return crosswalk.data.CWData(
        df,
        "obs",
        "obs_se",
        "alt_dorms",
        "ref_dorms",
        covs=[f"cov{i}" for i in range(num_covs)],
        study_id="study_id",
    )


@pytest.fixture
def cov_models():
    return [model.CovModel("intercept"), model.CovModel("cov0")]


@pytest.mark.parametrize("obs_type", ["diff_log", "diff_logit"])
def test_input(cwdata, obs_type, cov_models):
    _ = model.CWModel(cwdata, obs_type, cov_models=cov_models)


def test_design_mat(cwdata, cov_models):
    obs_type = "diff_log"
    cwmodel = model.CWModel(cwdata, obs_type, cov_models=cov_models)
    design_mat = cwmodel.design_mat
    assert np.allclose(cwmodel.relation_mat.sum(axis=1), 0.0)
    assert np.allclose(design_mat.sum(axis=1), 0.0)


@pytest.mark.parametrize("order_prior", [[["1", "2"], ["2", "3"]]])
def test_order_prior(cwdata, cov_models, order_prior):
    obs_type = "diff_log"
    cwmodel = model.CWModel(
        cwdata, obs_type, cov_models=cov_models, order_prior=order_prior
    )

    constraints_mat = cwmodel.constraint_mat
    assert isinstance(constraints_mat, np.ndarray)
    assert constraints_mat.shape == (4, cwmodel.num_vars)
    assert (constraints_mat.sum(axis=1) == 0.0).all()


@pytest.mark.parametrize("alt_dorm", ["2", "3"])
@pytest.mark.parametrize("ref_dorm", ["3"])
def test_adjust_orig_vals(cwdata, cov_models, alt_dorm, ref_dorm):
    obs_type = "diff_log"
    gold_dorm = ref_dorm
    cwmodel = model.CWModel(
        cwdata, obs_type, cov_models=cov_models, gold_dorm=gold_dorm
    )
    cwmodel.fit()
    new_df = pd.DataFrame(
        {
            "dorms": np.array([alt_dorm] * cwdata.num_obs),
            "vals": np.ones(cwdata.num_obs),
            "se": np.zeros(cwdata.num_obs),
        }
    )
    for cov in cwdata.covs.columns:
        new_df[cov] = np.ones(cwdata.num_obs)

    pred_df = cwmodel.adjust_orig_vals(
        new_df, orig_dorms="dorms", orig_vals_mean="vals", orig_vals_se="se"
    )
    assert np.allclose(
        pred_df["ref_vals_mean"],
        np.exp(
            np.sum(
                cwmodel.beta[cwmodel.var_idx[ref_dorm]]
                - cwmodel.beta[cwmodel.var_idx[alt_dorm]]
            )
        ),
    )
    assert np.allclose(pred_df["data_id"], np.arange(pred_df.shape[0]))


@pytest.mark.parametrize("cov_name", ["cov0", "cov1"])
@pytest.mark.parametrize("spline", [None, XSpline(np.linspace(-2.0, 2.0, 3), 3)])
@pytest.mark.parametrize("soln_name", [None, "cov"])
def test_cov_model(cwdata, cov_name, spline, soln_name):
    cov_model = model.CovModel(cov_name, spline=spline, soln_name=soln_name)

    if spline is None:
        assert cov_model.num_vars == 1
        assert (
            cov_model.create_design_mat(cwdata) == cwdata.covs[cov_name].values[:, None]
        ).all()
    else:
        assert cov_model.num_vars == spline.num_spline_bases - 1
        assert (
            cov_model.create_design_mat(cwdata)
            == spline.design_mat(cwdata.covs[cov_name])[:, 1:]
        ).all()

    if soln_name is None:
        assert cov_model.soln_name == cov_name
    else:
        assert cov_model.soln_name == soln_name


@pytest.mark.parametrize("cov_name", ["cov0", "cov1"])
@pytest.mark.parametrize("spline", [None, XSpline(np.linspace(-2.0, 2.0, 3), 3)])
@pytest.mark.parametrize("soln_name", [None, "cov"])
@pytest.mark.parametrize("spline_monotonicity", ["increasing", "decreasing", None])
@pytest.mark.parametrize("spline_convexity", ["convex", "concave", None])
def test_spline_prior(
    cov_name, spline, soln_name, spline_monotonicity, spline_convexity
):
    cov_model = model.CovModel(
        cov_name,
        spline=spline,
        spline_monotonicity=spline_monotonicity,
        spline_convexity=spline_convexity,
        soln_name=soln_name,
    )

    constraints_mat = cov_model.create_constraint_mat()

    if not cov_model.use_constraints:
        assert constraints_mat.size == 0
    else:
        num_constraints = (
            np.sum([spline_monotonicity is not None, spline_convexity is not None]) * 20
        )
        assert constraints_mat.shape == (num_constraints, cov_model.num_vars)


def test_pred_sd_full_covariance(cwdata, cov_models):
    gold = "3"
    cwmodel = model.CWModel(cwdata, "diff_log", cov_models=cov_models, gold_dorm=gold)
    cwmodel.fit()
    vcov = cwmodel.beta_vcov
    # covariance matrix is symmetric, matches beta_sd and is zero on the gold block
    assert np.allclose(vcov, vcov.T)
    assert np.allclose(np.sqrt(np.diag(vcov)), cwmodel.beta_sd)
    assert np.allclose(vcov[cwmodel.var_idx[gold], :], 0.0)

    # prediction sd equals sqrt(x' V x), including the covariances
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, cwmodel.num_vars))
    expected = np.sqrt(np.einsum("ij,jk,ik->i", x, vcov, x))
    assert np.allclose(cwmodel.get_pred_sd(x), expected)

    # adjust_orig_vals uses it: covariate far from zero, one row per dorm
    alts = [d for d in cwmodel.cwdata.unique_dorms if d != gold]
    dorms = alts + [gold]
    new_df = pd.DataFrame(
        {"dorms": dorms, "vals": 0.5, "se": 0.1, "cov0": 10.0, "cov1": 0.0, "cov2": 0.0}
    )
    pred = cwmodel.adjust_orig_vals(
        new_df, orig_dorms="dorms", orig_vals_mean="vals", orig_vals_se="se"
    )
    x_new = np.zeros((len(dorms), cwmodel.num_vars))
    for i, dorm in enumerate(alts):
        x_new[i, cwmodel.var_idx[dorm]] = [1.0, 10.0]
        x_new[i, cwmodel.var_idx[gold]] = [-1.0, -10.0]
    expected = np.sqrt(np.einsum("ij,jk,ik->i", x_new, vcov, x_new))
    assert np.allclose(pred["pred_diff_sd"], expected)
    assert pred["pred_diff_sd"].iloc[-1] == 0.0


def test_fit_does_not_depend_on_row_order():
    rng = np.random.default_rng(0)
    num_obs = 60
    df = pd.DataFrame(
        {
            "obs_se": 0.1 + rng.random(num_obs) * 0.1,
            "alt_dorms": "alt",
            "ref_dorms": "ref",
            "study_id": rng.integers(0, 10, num_obs),
            "x": rng.uniform(0.0, 4.0, num_obs),
        }
    )
    df["obs"] = 0.5 + 1.0 * df["x"] + rng.normal(0.0, 0.1, num_obs)

    def fit_beta(frame):
        cwdata = crosswalk.data.CWData(
            frame,
            "obs",
            "obs_se",
            "alt_dorms",
            "ref_dorms",
            covs=["x"],
            study_id="study_id",
        )
        cwmodel = model.CWModel(
            cwdata,
            "diff_log",
            cov_models=[model.CovModel("intercept"), model.CovModel("x")],
            gold_dorm="ref",
        )
        cwmodel.fit()
        return cwmodel.fixed_vars["alt"]

    beta_sorted = fit_beta(
        df.sort_values("study_id", kind="stable").reset_index(drop=True)
    )
    beta_unsorted = fit_beta(df)
    beta_shuffled = fit_beta(df.sample(frac=1.0, random_state=1))

    assert np.allclose(beta_sorted, [0.5, 1.0], atol=0.1)
    assert np.allclose(beta_unsorted, beta_sorted, atol=1e-6)
    assert np.allclose(beta_shuffled, beta_sorted, atol=1e-6)


@pytest.mark.parametrize("use_study_id", [False, True])
def test_adjust_orig_vals_ref_dorms(use_study_id):
    rng = np.random.default_rng(0)
    num_obs = 300
    df = pd.DataFrame(
        {
            "alt_dorms": rng.choice(["B", "C"], num_obs),
            "ref_dorms": rng.choice(["A", "B", "C"], num_obs),
            "obs_se": 0.3,
            "study_id": rng.integers(0, 20, num_obs),
        }
    )
    df = df[df["alt_dorms"] != df["ref_dorms"]]
    df = df.sort_values("study_id", kind="stable").reset_index(drop=True)
    effect = {"A": 0.0, "B": 1.0, "C": 2.0}
    study_effect = rng.normal(0.0, 0.5, 20)
    df["obs"] = (
        df["alt_dorms"].map(effect)
        - df["ref_dorms"].map(effect)
        + study_effect[df["study_id"]]
        + rng.normal(0.0, 0.3, len(df))
    )
    cwdata = crosswalk.data.CWData(
        df, "obs", "obs_se", "alt_dorms", "ref_dorms", study_id="study_id"
    )
    cwmodel = model.CWModel(
        cwdata,
        "diff_logit",
        cov_models=[model.CovModel("intercept")],
        gold_dorm="A",
    )
    cwmodel.fit()

    new_df = pd.DataFrame(
        {
            "dorms": ["A", "B", "C", "A", "B", "C"],
            "target": ["A", "A", "A", "C", "C", "C"],
            "vals": 0.3,
            "se": 0.05,
            "study_id": cwdata.unique_study_id[0],
        }
    )
    kwargs = dict(
        orig_dorms="dorms",
        orig_vals_mean="vals",
        orig_vals_se="se",
        study_id="study_id" if use_study_id else None,
        ref_dorms="target",
    )
    if use_study_id:
        with pytest.warns(UserWarning, match="without random effects"):
            pred = cwmodel.adjust_orig_vals(new_df, **kwargs)
    else:
        pred = cwmodel.adjust_orig_vals(new_df, **kwargs)
    no_change = (new_df["dorms"] == new_df["target"]).to_numpy()
    changed = ~no_change

    # rows already in their target dorm are returned unchanged
    assert np.allclose(pred["ref_vals_mean"][no_change], 0.3)
    assert np.allclose(pred["ref_vals_sd"][no_change], 0.05)
    assert np.allclose(pred["pred_diff_sd"][no_change], 0.0)

    # every other row, including the gold row converted to C, gets the
    # prediction uncertainty and gamma: the SE is larger in logit space
    assert (pred["pred_diff_sd"][changed] > 0.0).all()
    _, se_logit = crosswalk.utils.linear_to_logit(
        pred["ref_vals_mean"].to_numpy(), pred["ref_vals_sd"].to_numpy()
    )
    _, orig_se_logit = crosswalk.utils.linear_to_logit(
        np.array([0.3]), np.array([0.05])
    )
    assert (se_logit[changed] > orig_se_logit[0]).all()

    # the gold row converted to C gets no random effect, with or without study_id
    pred_no_re = cwmodel.adjust_orig_vals(
        new_df,
        orig_dorms="dorms",
        orig_vals_mean="vals",
        orig_vals_se="se",
        ref_dorms="target",
    )
    assert np.isclose(pred["ref_vals_mean"][3], pred_no_re["ref_vals_mean"][3])


def test_adjust_orig_vals_study_id_to_gold_no_warning(cwdata, cov_models):
    cwmodel = model.CWModel(cwdata, "diff_log", cov_models=cov_models, gold_dorm="3")
    cwmodel.fit()
    new_df = pd.DataFrame(
        {
            "dorms": ["0", "3"],
            "vals": 0.5,
            "se": 0.1,
            "cov0": 0.0,
            "cov1": 0.0,
            "cov2": 0.0,
            "study_id": cwdata.unique_study_id[0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cwmodel.adjust_orig_vals(
            new_df,
            orig_dorms="dorms",
            orig_vals_mean="vals",
            orig_vals_se="se",
            study_id="study_id",
        )
