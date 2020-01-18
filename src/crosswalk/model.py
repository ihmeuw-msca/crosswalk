# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
import numpy as np
import warnings
from limetr import LimeTr
from xspline import XSpline
from . import data
from . import utils


class CovModel:
    """Covariate model.
    """
    def __init__(self, cov_name, spline=None, soln_name=None):
        """Constructor of the CovModel.

        Args:
            cov_name(str):
                Corresponding covariate name.
            spline (XSpline | None, optional):
                If using spline, passing in spline object.
            soln_name (str):
                Name of the corresponding covariates multiplier.
        """
        # check the input
        assert isinstance(cov_name, str)
        assert isinstance(spline, XSpline) or spline is None
        assert isinstance(soln_name, str) or soln_name is None

        self.cov_name = cov_name
        self.spline = spline
        self.use_spline = spline is not None
        self.soln_name = cov_name if soln_name is None else soln_name

        if self.use_spline:
            self.num_vars = spline.num_spline_bases - 1
        else:
            self.num_vars = 1

    def create_design_mat(self, cwdata):
        """Create design matrix.

        Args:
            cwdata (crosswalk.CWData):
                Data structure has all the information.
        """
        assert self.cov_name in cwdata.covs, "Unkown covariates, not appear" \
                                             "in the data."
        cov = cwdata.covs[self.cov_name]
        if self.use_spline:
            mat = self.spline.design_mat(cov)[:, 1:]
        else:
            mat = cov[:, None]
        return mat


class CWModel:
    """Cross Walk model.
    """
    def __init__(self, cwdata,
                 obs_type='diff_log',
                 dorm_models=None,
                 diff_models=None,
                 gold_dorm=None,
                 dorm_order_prior=None):
        """Constructor of CWModel.
        Args:
            cwdata (data.CWData):
                Data for cross walk.
            obs_type (str, optional):
                Type of observation can only be chosen from `'diff_log'` and
                `'diff_logit'`.
            dorm_models (list{crosswalk.CovModel}):
                A list of covariate models for the definitions/methods
            diff_models (list{crosswalk.CovModel}):
                A list of covaraite models for the differences
            gold_dorm (str | None, optional):
                Gold standard definition/method.
            dorm_order_prior (list{list{str}} | None, optional):
                Order priors between different definitions.
        """
        self.cwdata = cwdata
        self.obs_type = obs_type
        self.dorm_models = utils.default_input(dorm_models,
                                               [CovModel('intercept')])
        self.diff_models = utils.default_input(diff_models, [])
        self.gold_dorm = utils.default_input(gold_dorm, cwdata.max_ref_dorm)
        self.dorm_order_prior = dorm_order_prior

        # check input
        self.check()

        # create function for prediction
        if self.obs_type == 'diff_log':
            def obs_fun(x):
                return np.log(x)

            def obs_inv_fun(y):
                return np.exp(y)
        else:
            def obs_fun(x):
                return np.log(x/(1.0 - x))

            def obs_inv_fun(y):
                return 1.0/(1.0 + np.exp(-y))

        self.obs_fun = obs_fun
        self.obs_inv_fun = obs_inv_fun

        # variable names
        self.dorm_vars = ['_'.join(['dorm', dorm, model.soln_name])
                          for dorm in self.cwdata.unique_dorms
                          for model in self.dorm_models]
        self.diff_vars = ['_'.join(['diff', model.soln_name])
                          for model in self.diff_models]
        self.vars = [dorm for dorm in self.cwdata.unique_dorms] + ['diff']

        # dimensions
        self.num_vars_per_dorm = sum([model.num_vars
                                      for model in self.dorm_models])
        self.num_dorm_vars = self.num_vars_per_dorm*self.cwdata.num_dorms
        self.num_diff_vars = sum([model.num_vars
                                  for model in self.diff_models])
        self.num_vars = self.num_dorm_vars + self.num_diff_vars

        # indices for easy access the variables
        var_sizes = np.array([self.num_vars_per_dorm]*self.cwdata.num_dorms +
                             [self.num_diff_vars])
        var_idx = utils.sizes_to_indices(var_sizes)
        self.var_idx = {
            var: var_idx[i]
            for i, var in enumerate(self.vars)
        }

        # create design matrix
        self.design_mat = self.create_design_mat()
        self.constraint_mat = self.create_constraint_mat()

        # place holder for the solutions
        self.beta = None
        self.gamma = None
        self.fixed_vars = None
        self.random_vars = None

    def check(self):
        """Check input type, dimension and values.
        """
        assert isinstance(self.cwdata, data.CWData)
        assert self.obs_type in ['diff_log', 'diff_logit'], "Unsupport " \
                                                            "observation type"
        assert isinstance(self.dorm_models, list)
        assert isinstance(self.diff_models, list)
        assert all([isinstance(model, CovModel) for model in self.dorm_models])
        assert all([isinstance(model, CovModel) for model in self.diff_models])
        assert all([not model.use_spline for model in self.dorm_models]), \
            "Do not support using spline in definitions/methods model."

        assert self.gold_dorm in self.cwdata.unique_dorms

        assert self.dorm_order_prior is None or \
               isinstance(self.dorm_order_prior, list)

    @property
    def relation_mat(self):
        """Creating relation matrix.
        Returns:
            numpy.ndarray:
                Returns relation matrix with 1 encode alternative definition
                and -1 encode reference definition.
        """

        relation_mat = np.zeros((self.cwdata.num_obs, self.cwdata.num_dorms))
        relation_mat[range(self.cwdata.num_obs),
                     [self.cwdata.dorm_idx[dorm]
                      for dorm in self.cwdata.alt_dorms]] = 1.0
        relation_mat[range(self.cwdata.num_obs),
                     [self.cwdata.dorm_idx[dorm]
                      for dorm in self.cwdata.ref_dorms]] = -1.0

        return relation_mat

    @property
    def dorm_cov_mat(self):
        """Create covariates matrix for definitions/methods model.

        Returns:
            numpy.ndarray:
                Returns covarites matrix.
        """
        return np.hstack([model.create_design_mat(self.cwdata)
                          for model in self.dorm_models])

    @property
    def diff_cov_mat(self):
        """Create covariates matrix for difference.

        Returns:
            numpy.ndarray:
                Returns covarites matrix.
        """
        if self.diff_models:
            return np.hstack([model.create_design_mat(self.cwdata)
                              for model in self.diff_models])
        else:
            return np.array([]).reshape(self.cwdata.num_obs, 0)

    def create_design_mat(self):
        """Create linear design matrix.
        Returns:
            numpy.ndarray:
                Returns linear design matrix.
        """
        mat = (
            self.relation_mat.ravel()[:, None] *
            np.repeat(self.dorm_cov_mat, self.cwdata.num_dorms, axis=0)
        ).reshape(self.cwdata.num_obs, self.num_dorm_vars)
        mat = np.hstack((mat, self.diff_cov_mat))

        return mat

    def create_constraint_mat(self):
        """Create constraint matrix.

        Returns:
            numpy.ndarray:
                Return constraints matrix.
        """
        if self.dorm_order_prior is None:
            return None

        mat = []
        dorm_cov_mat = self.dorm_cov_mat
        min_dorm_cov_mat = np.min(dorm_cov_mat, axis=0)
        max_dorm_cov_mat = np.max(dorm_cov_mat, axis=0)

        if np.allclose(min_dorm_cov_mat, max_dorm_cov_mat):
            design_mat = min_dorm_cov_mat[None, :]
        else:
            design_mat = np.vstack((
                min_dorm_cov_mat,
                max_dorm_cov_mat
            ))
        for p in self.dorm_order_prior:
            sub_mat = np.zeros((design_mat.shape[0], self.num_vars))
            sub_mat[:, self.var_idx[p[0]]] = design_mat
            sub_mat[:, self.var_idx[p[1]]] = -design_mat
            mat.append(sub_mat)

        return np.vstack(mat)

    def fit(self, max_iter=100):
        """Optimize the model parameters.
        This is a interface to limetr.
        Args:
            max_iter (int, optional):
                Maximum number of iterations.
        """
        # dimensions for limetr
        n = self.cwdata.study_sizes
        k_beta = self.num_vars
        k_gamma = 1
        y = self.cwdata.obs
        s = self.cwdata.obs_se
        x = self.design_mat
        z = np.ones((self.cwdata.num_obs, 1))

        uprior = np.array([[-np.inf]*self.num_vars + [0.0],
                           [np.inf]*self.num_vars + [np.inf]])
        uprior[:, self.var_idx[self.gold_dorm]] = 0.0

        if self.constraint_mat is None:
            cfun = None
            jcfun = None
            cvec = None
        else:
            num_constraints = self.constraint_mat.shape[0]
            cmat = np.hstack((self.constraint_mat,
                              np.zeros((num_constraints, 1))))

            cvec = np.array([[-np.inf]*num_constraints,
                             [0.0]*num_constraints])

            def cfun(var):
                return cmat.dot(var)

            def jcfun(var):
                return cmat

        def fun(var):
            return x.dot(var)

        def jfun(beta):
            return x

        lt = LimeTr(n, k_beta, k_gamma, y, fun, jfun, z,
                    S=s,
                    uprior=uprior,
                    C=cfun,
                    JC=jcfun,
                    c=cvec)
        self.beta, self.gamma, _ = lt.fitModel(inner_print_level=5,
                                               inner_max_iter=max_iter)

        self.fixed_vars = {
            var: self.beta[self.var_idx[var]]
            for var in self.vars
        }
        self.random_vars = self.gamma

    def predict_alt_vals(self, ref_vals):
        """Predict the alternative definitions/methods values.

        Args:
            ref_vals (numpy.ndarray):
                Reference definitions/methods values.

        Returns:
            numpy.ndarray:
                Return the corrected alternative definitions/methods values.
        """
        # compute the differences
        diff = self.design_mat.dot(self.beta)
        alt_vals = self.obs_inv_fun(diff + self.obs_fun(ref_vals))

        return alt_vals
