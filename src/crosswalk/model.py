# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
import numpy as np
import scipy.linalg as splinalg
from limetr import LimeTr
from xspline import XSpline
from . import data
from . import utils


class CovModel:
    """Covariate model.
    """
    def __init__(self, cov_name,
                 spline=None,
                 spline_monotonicity=None,
                 spline_convexity=None,
                 soln_name=None):
        """Constructor of the CovModel.

        Args:
            cov_name(str):
                Corresponding covariate name.
            spline (XSpline | None, optional):
                If using spline, passing in spline object.
            spline_monotonicity (str | None, optional):
                Spline shape prior, indicate if spline is increasing or
                decreasing.
            spline_convexity (str | None, optional):
                Spline shape prior, indicate if spline is convex or concave.
            soln_name (str):
                Name of the corresponding covariates multiplier.
        """
        # check the input
        assert isinstance(cov_name, str)
        assert isinstance(spline, XSpline) or spline is None
        if spline_monotonicity is not None:
            assert spline_monotonicity in ['increasing', 'decreasing']
        if spline_convexity is not None:
            assert spline_convexity in ['convex', 'concave']
        assert isinstance(soln_name, str) or soln_name is None

        self.cov_name = cov_name
        self.spline = spline
        self.spline_monotonicity = spline_monotonicity
        self.spline_convexity = spline_convexity
        self.use_spline = spline is not None
        self.use_constraints = self.use_spline and (
                self.spline_monotonicity is not None or
                self.spline_convexity is not None
        )
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

        Returns:
            numpy.ndarray:
                Return the design matrix from linear cov or spline.
        """
        assert self.cov_name in cwdata.covs.columns,\
            "Unkown covariates, not appear in the data."
        cov = cwdata.covs[self.cov_name].values
        if self.use_spline:
            mat = self.spline.design_mat(cov)[:, 1:]
        else:
            mat = cov[:, None]
        return mat

    def create_constraint_mat(self, num_points=20):
        """Create constraints matrix.

        Args:
            num_points (int, optional):
                Number of approximation points to cover the interval for spline.

        Returns:
            numpy.ndarray:
                Return constraints matrix if have any.
        """
        mat = np.array([]).reshape(0, self.num_vars)
        if not self.use_constraints:
            return mat
        points = np.linspace(self.spline.knots[0],
                             self.spline.knots[-1],
                             num_points)

        if self.spline_monotonicity is not None:
            sign = 1.0 if self.spline_monotonicity is 'decreasing' else -1.0
            mat = np.vstack((mat,
                             sign*self.spline.design_dmat(points, 1)[:, 1:]))

        if self.spline_convexity is not None:
            sign = 1.0 if self.spline_convexity is 'concave' else -1.0
            mat = np.vstack((mat,
                             sign*self.spline.design_dmat(points, 2)[:, 1:]))

        return mat


class CWModel:
    """Cross Walk model.
    """
    def __init__(self, cwdata,
                 obs_type='diff_log',
                 dorm_models=None,
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
            gold_dorm (str | None, optional):
                Gold standard definition/method.
            dorm_order_prior (list{list{str}} | None, optional):
                Order priors between different definitions.
        """
        self.cwdata = cwdata
        self.obs_type = obs_type
        self.dorm_models = utils.default_input(dorm_models,
                                               [CovModel('intercept')])
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
        self.vars = [dorm for dorm in self.cwdata.unique_dorms]

        # dimensions
        self.num_vars_per_dorm = sum([model.num_vars
                                      for model in self.dorm_models])
        self.num_vars = self.num_vars_per_dorm*self.cwdata.num_dorms

        # indices for easy access the variables
        var_sizes = np.array([self.num_vars_per_dorm]*self.cwdata.num_dorms)
        var_idx = utils.sizes_to_indices(var_sizes)
        self.var_idx = {
            var: var_idx[i]
            for i, var in enumerate(self.vars)
        }

        # create design matrix
        self.relation_mat = self.create_relation_mat()
        self.dorm_cov_mat = self.create_dorm_cov_mat()
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
        assert all([isinstance(model, CovModel) for model in self.dorm_models])
        assert all([not model.use_spline for model in self.dorm_models]), \
            "Do not support using spline in definitions/methods model."

        assert self.gold_dorm in self.cwdata.unique_dorms

        assert self.dorm_order_prior is None or \
               isinstance(self.dorm_order_prior, list)

    def create_relation_mat(self, cwdata=None):
        """Creating relation matrix.

        Args:
            cwdata (data.CWData | None, optional):
                Optional data set, if None, use `self.cwdata`.

        Returns:
            numpy.ndarray:
                Returns relation matrix with 1 encode alternative definition
                and -1 encode reference definition.
        """
        cwdata = utils.default_input(cwdata,
                                     default=self.cwdata)
        assert isinstance(cwdata, data.CWData)

        relation_mat = np.zeros((cwdata.num_obs, cwdata.num_dorms))
        relation_mat[range(cwdata.num_obs),
                     [cwdata.dorm_idx[dorm]
                      for dorm in cwdata.alt_dorms]] = 1.0
        relation_mat[range(cwdata.num_obs),
                     [cwdata.dorm_idx[dorm]
                      for dorm in cwdata.ref_dorms]] = -1.0

        return relation_mat

    def create_dorm_cov_mat(self, cwdata=None):
        """Create covariates matrix for definitions/methods model.

        Args:
            cwdata (data.CWData | None, optional):
                Optional data set, if None, use `self.cwdata`.

        Returns:
            numpy.ndarray:
                Returns covarites matrix.
        """
        cwdata = utils.default_input(cwdata,
                                     default=self.cwdata)
        assert isinstance(cwdata, data.CWData)

        return np.hstack([model.create_design_mat(cwdata)
                          for model in self.dorm_models])

    def create_design_mat(self,
                          cwdata=None,
                          relation_mat=None,
                          dorm_cov_mat=None):
        """Create linear design matrix.

        Args:
            cwdata (data.CWData | None, optional):
                Optional data set, if None, use `self.cwdata`.

        Returns:
            numpy.ndarray:
                Returns linear design matrix.
        """
        cwdata = utils.default_input(cwdata,
                                     default=self.cwdata)
        relation_mat = utils.default_input(relation_mat,
                                           default=self.relation_mat)
        dorm_cov_mat = utils.default_input(dorm_cov_mat,
                                           default=self.dorm_cov_mat)

        mat = (
            relation_mat.ravel()[:, None] *
            np.repeat(dorm_cov_mat, cwdata.num_dorms, axis=0)
        ).reshape(cwdata.num_obs, self.num_vars)

        return mat

    def create_constraint_mat(self):
        """Create constraint matrix.

        Returns:
            numpy.ndarray:
                Return constraints matrix.
        """
        mat = np.array([]).reshape(0, self.num_vars)
        if self.dorm_order_prior is not None:
            dorm_constraint_mat = []
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
                dorm_constraint_mat.append(sub_mat)
            dorm_constraint_mat = np.vstack(dorm_constraint_mat)
            mat = np.vstack((mat, dorm_constraint_mat))

        if mat.size == 0:
            return None
        else:
            return mat


    def fit(self, max_iter=100, inlier_pct=1.0):
        """Optimize the model parameters.
        This is a interface to limetr.
        Args:
            max_iter (int, optional):
                Maximum number of iterations.
            inlier_pct (float, optional):
                How much percentage of the data do you trust.
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
                    c=cvec,
                    inlier_percentage=inlier_pct)
        self.beta, self.gamma, _ = lt.fitModel(inner_print_level=5,
                                               inner_max_iter=max_iter)

        self.fixed_vars = {
            var: self.beta[self.var_idx[var]]
            for var in self.vars
        }
        self.random_vars = self.gamma

    def adjust_alt_vals(self, df, alt_dorms, alt_vals):
        """Adjust alternative values.

        Args:
            df (pd.DataFrame):
                Data frame of the alternative values that need to be adjusted.
            alt_dorms (str):
                Name of the column in `df` that contains the alternative
                definitions or methods.
            alt_vals (str):
                Name of the column in `df` that contains the alternative values.

        Returns:
            numpy.ndarray:
                The adjusted alternative values.
        """
        df_copy = df.copy()
        ref_dorms = 'ref_dorms'
        df_copy[ref_dorms] = np.array([self.gold_dorm]*df_copy.shape[0])
        new_cwdata = data.CWData(df_copy,
                                 alt_dorms=alt_dorms,
                                 ref_dorms=ref_dorms,
                                 covs=list(self.cwdata.covs.columns))

        # transfer data dorm structure to the new_cwdata
        new_cwdata.copy_dorm_structure(self.cwdata)

        # create new design matrix
        new_relation_mat = self.create_relation_mat(cwdata=new_cwdata)
        new_dorm_cov_mat = self.create_dorm_cov_mat(cwdata=new_cwdata)
        new_design_mat = self.create_design_mat(cwdata=new_cwdata,
                                                relation_mat=new_relation_mat,
                                                dorm_cov_mat=new_dorm_cov_mat)

        # compute the corresponding gold_dorm value
        ref_vals = self.obs_inv_fun(self.obs_fun(df[alt_vals].values) -
                                    new_design_mat.dot(self.beta))

        return ref_vals
