# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
import warnings
import numpy as np
import pandas as pd
import limetr
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
            sign = 1.0 if self.spline_monotonicity == 'decreasing' else -1.0
            mat = np.vstack((mat,
                             sign*self.spline.design_dmat(points, 1)[:, 1:]))

        if self.spline_convexity is not None:
            sign = 1.0 if self.spline_convexity == 'concave' else -1.0
            mat = np.vstack((mat,
                             sign*self.spline.design_dmat(points, 2)[:, 1:]))

        return mat


class CWModel:
    """Cross Walk model.
    """
    def __init__(self, cwdata,
                 obs_type='diff_log',
                 cov_models=None,
                 gold_dorm=None,
                 order_prior=None,
                 use_random_intercept=True):
        """Constructor of CWModel.
        Args:
            cwdata (data.CWData):
                Data for cross walk.
            obs_type (str, optional):
                Type of observation can only be chosen from `'diff_log'` and
                `'diff_logit'`.
            cov_models (list{crosswalk.CovModel}):
                A list of covariate models for the definitions/methods
            gold_dorm (str | None, optional):
                Gold standard definition/method.
            order_prior (list{list{str}} | None, optional):
                Order priors between different definitions.
            use_random_intercept (bool, optional):
                If ``True``, use random intercept.
        """
        self.cwdata = cwdata
        self.obs_type = obs_type
        self.cov_models = utils.default_input(cov_models,
                                              [CovModel('intercept')])
        self.gold_dorm = utils.default_input(gold_dorm, cwdata.max_ref_dorm)
        self.order_prior = order_prior
        self.use_random_intercept = use_random_intercept
        if self.cwdata.num_studies == 0 and self.use_random_intercept:
            warnings.warn("Must have study_id to use random intercept."
                          " Reset use_random_intercept to False.")
            self.use_random_intercept = False

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
                                      for model in self.cov_models])
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
        self.cov_mat = self.create_cov_mat()
        self.design_mat = self.create_design_mat()
        self.constraint_mat = self.create_constraint_mat()

        # place holder for the solutions
        self.beta = None
        self.beta_sd = None
        self.gamma = None
        self.fixed_vars = None
        self.random_vars = None

    def check(self):
        """Check input type, dimension and values.
        """
        assert isinstance(self.cwdata, data.CWData)
        assert self.obs_type in ['diff_log', 'diff_logit'], \
            "Unsupport observation type"
        assert isinstance(self.cov_models, list)
        assert all([isinstance(model, CovModel) for model in self.cov_models])

        assert self.gold_dorm in self.cwdata.unique_dorms

        assert self.order_prior is None or isinstance(self.order_prior, list)

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
        for i, dorms in enumerate(cwdata.alt_dorms):
            for dorm in dorms:
                relation_mat[i, cwdata.dorm_idx[dorm]] += 1.0

        for i, dorms in enumerate(cwdata.ref_dorms):
            for dorm in dorms:
                relation_mat[i, cwdata.dorm_idx[dorm]] -= 1.0

        return relation_mat

    def create_cov_mat(self, cwdata=None):
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
                          for model in self.cov_models])

    def create_design_mat(self,
                          cwdata=None,
                          relation_mat=None,
                          cov_mat=None):
        """Create linear design matrix.

        Args:
            cwdata (data.CWData | None, optional):
                Optional data set, if None, use `self.cwdata`.
            relation_mat (numpy.ndarray | None, optional):
                Optional relation matrix, if None, use `self.relation_mat`
            cov_mat (numpy.ndarray | None, optional):
                Optional covariates matrix, if None, use `self.cov_mat`

        Returns:
            numpy.ndarray:
                Returns linear design matrix.
        """
        cwdata = utils.default_input(cwdata,
                                     default=self.cwdata)
        relation_mat = utils.default_input(relation_mat,
                                           default=self.relation_mat)
        cov_mat = utils.default_input(cov_mat,
                                           default=self.cov_mat)

        mat = (
            relation_mat.ravel()[:, None] *
            np.repeat(cov_mat, cwdata.num_dorms, axis=0)
        ).reshape(cwdata.num_obs, self.num_vars)

        return mat

    def create_constraint_mat(self):
        """Create constraint matrix.

        Returns:
            numpy.ndarray:
                Return constraints matrix.
        """
        mat = np.array([]).reshape(0, self.num_vars)
        if self.order_prior is not None:
            dorm_constraint_mat = []
            cov_mat = self.cov_mat
            min_cov_mat = np.min(cov_mat, axis=0)
            max_cov_mat = np.max(cov_mat, axis=0)

            if np.allclose(min_cov_mat, max_cov_mat):
                design_mat = min_cov_mat[None, :]
            else:
                design_mat = np.vstack((
                    min_cov_mat,
                    max_cov_mat
                ))
            for p in self.order_prior:
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
        if n.size == 0:
            n = np.full(self.cwdata.num_obs, 1)
        k_beta = self.num_vars
        k_gamma = 1
        y = self.cwdata.obs
        s = self.cwdata.obs_se
        x = self.design_mat
        z = np.ones((self.cwdata.num_obs, 1))

        uprior = np.array([[-np.inf]*self.num_vars,
                           [np.inf]*self.num_vars])
        uprior[:, self.var_idx[self.gold_dorm]] = 0.0
        if self.use_random_intercept:
            uprior = np.hstack((uprior, np.array([[0.0], [np.inf]])))
        else:
            uprior = np.hstack((uprior, np.array([[0.0], [0.0]])))

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

        self.lt = LimeTr(n, k_beta, k_gamma, y, fun, jfun, z,
                    S=s,
                    uprior=uprior,
                    C=cfun,
                    JC=jcfun,
                    c=cvec,
                    inlier_percentage=inlier_pct)
        self.beta, self.gamma, self.w = self.lt.fitModel(inner_print_level=5,
                                                         inner_max_iter=max_iter)

        self.fixed_vars = {
            var: self.beta[self.var_idx[var]]
            for var in self.vars
        }
        if self.use_random_intercept:
            u = self.lt.estimateRE()
            self.random_vars = {
                sid: u[i]
                for i, sid in enumerate(self.cwdata.unique_study_id)
            }
        else:
            self.random_vars = dict()

        # compute the posterior distribution of beta
        x = self.lt.JF(self.lt.beta)*np.sqrt(self.lt.w)[:, None]
        z = self.lt.Z*np.sqrt(self.lt.w)[:, None]
        v = limetr.utils.VarMat(self.lt.V**self.lt.w, z, self.lt.gamma, self.lt.n)

        if hasattr(self.lt, 'gprior'):
            beta_gprior_sd = self.lt.gprior[:, self.lt.idx_beta][1]
        else:
            beta_gprior_sd = np.repeat(np.inf, self.lt.k_beta)

        unconstrained_id = np.hstack([
            np.arange(self.lt.k_beta)[self.var_idx[dorm]]
            for dorm in self.cwdata.unique_dorms
            if dorm != self.gold_dorm
        ])

        hessian = x.T.dot(v.invDot(x)) + np.diag(1.0/beta_gprior_sd**2)
        hessian = np.delete(hessian, self.var_idx[self.gold_dorm], axis=0)
        hessian = np.delete(hessian, self.var_idx[self.gold_dorm], axis=1)

        self.beta_sd = np.zeros(self.lt.k_beta)
        self.beta_sd[unconstrained_id] = np.sqrt(np.diag(
            np.linalg.inv(hessian)
        ))

    def create_result_df(self) -> pd.DataFrame:
        """Create result data frame.

        Returns:
            pd.DataFrame: Data frame that contains the result.
        """
        # column of dorms
        dorms = np.repeat(self.cwdata.unique_dorms, self.num_vars_per_dorm)
        # column of covariate name
        cov_names = []
        for model in self.cov_models:
            if model.spline is None:
                cov_names.append(model.cov_name)
            else:
                cov_names.extend([f'{model.cov_name}_spline_{i}' for i in range(model.num_vars)])
        cov_names *= self.cwdata.num_dorms

        # create data frame
        df = pd.DataFrame({
            'dorms': dorms,
            'cov_names': cov_names,
            'beta': self.beta,
            'beta_sd': self.beta_sd,
        })
        if self.use_random_intercept:
            gamma = np.hstack((self.lt.gamma, np.full(self.num_vars - 1, np.nan)))
            re = np.hstack((self.lt.u, np.full((self.cwdata.num_studies, self.num_vars - 1), np.nan)))
            df['gamma'] = gamma
            for i, study_id in enumerate(self.cwdata.unique_study_id):
                df[study_id] = re[i]

        return df

    def save_result_df(self, folder: str, filename: str = 'result.csv'):
        """Save result.

        Args:
            folder (str): Path to the result folder.
            filename (str): Name of the result. Default to `'result.csv'`.
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        df = self.create_result_df()
        df.to_csv(folder + '/' + filename, index=False)

    def adjust_orig_vals(self, df,
                         orig_dorms,
                         orig_vals_mean,
                         orig_vals_se,
                         study_id=None,
                         data_id=None):
        """Adjust alternative values.

        Args:
            df (pd.DataFrame):
                Data frame of the alternative values that need to be adjusted.
            orig_dorms (str):
                Name of the column in `df` that contains the alternative
                definitions or methods.
            orig_vals_mean (str):
                Name of the column in `df` that contains the alternative values.
            orig_vals_se (str):
                Name of the column in `df` that contains the standard error of
                alternative values.
            study_id (str | None, optional):
                If not `None`, predict with the random effects.
            data_id (str | None, optional):
                If `None` create data_id by the integer sequence.

        Returns:
            pandas.DataFrame:
                The adjusted values and standard deviations.
        """
        df_copy = df.copy()
        ref_dorms = 'ref_dorms'
        df_copy[ref_dorms] = np.array([self.gold_dorm]*df_copy.shape[0])
        if 'intercept' not in df_copy.columns:
            df_copy['intercept'] = np.ones(df_copy.shape[0])
        new_cwdata = data.CWData(df_copy,
                                 alt_dorms=orig_dorms,
                                 ref_dorms=ref_dorms,
                                 dorm_separator=self.cwdata.dorm_separator,
                                 covs=list(self.cwdata.covs.columns),
                                 data_id=data_id,
                                 add_intercept=False)

        # transfer data dorm structure to the new_cwdata
        new_cwdata.copy_dorm_structure(self.cwdata)

        # create new design matrix
        new_relation_mat = self.create_relation_mat(cwdata=new_cwdata)
        new_cov_mat = self.create_cov_mat(cwdata=new_cwdata)
        new_design_mat = self.create_design_mat(cwdata=new_cwdata,
                                                relation_mat=new_relation_mat,
                                                cov_mat=new_cov_mat)

        # calculate the random effects
        if study_id is not None:
            random_effects = np.array([
                self.random_vars[sid]
                if sid in self.random_vars else 0.0
                for sid in df[study_id]
            ])
        else:
            random_effects = np.zeros(df.shape[0])
        random_effects[df[orig_dorms].values == self.gold_dorm] = 0.0

        # compute the corresponding gold_dorm value
        if self.obs_type == 'diff_log':
            transformed_orig_vals_mean,\
            transformed_orig_vals_se = utils.linear_to_log(
                df[orig_vals_mean].values,
                df[orig_vals_se].values
            )
        else:
            transformed_orig_vals_mean, \
            transformed_orig_vals_se = utils.linear_to_logit(
                df[orig_vals_mean].values,
                df[orig_vals_se].values
            )

        pred_diff_mean = new_design_mat.dot(self.beta)
        pred_diff_sd = np.sqrt(np.array([
            (new_design_mat[i]**2).dot(self.beta_sd**2)
            if dorm != self.gold_dorm else 0.0
            for i, dorm in enumerate(df[orig_dorms])
        ]))

        transformed_ref_vals_mean = transformed_orig_vals_mean - \
            pred_diff_mean - random_effects
        transformed_ref_vals_sd = np.sqrt(transformed_orig_vals_se**2 +
                                          pred_diff_sd**2 + self.gamma[0]**2)

        if self.obs_type == 'diff_log':
            ref_vals_mean,\
            ref_vals_sd = utils.log_to_linear(transformed_ref_vals_mean,
                                              transformed_ref_vals_sd)
        else:
            ref_vals_mean,\
            ref_vals_sd = utils.logit_to_linear(transformed_ref_vals_mean,
                                                transformed_ref_vals_sd)

        pred_df = pd.DataFrame({
            'ref_vals_mean': ref_vals_mean,
            'ref_vals_sd': ref_vals_sd,
            'pred_diff_mean': pred_diff_mean,
            'pred_diff_sd': pred_diff_sd,
            'data_id': new_cwdata.data_id
        })

        return pred_df
