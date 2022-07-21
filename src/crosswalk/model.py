import warnings
from typing import Dict, List, Optional

import limetr
import numpy as np
import pandas as pd
from anml.getter.spline import SplineGetter
from limetr import LimeTr
from pandas import DataFrame

from . import data, utils


class CovModel:
    """Covariate model.

    Parameters
    ----------
    cov_name
        Corresponding covariate name.
    spline_settings
        A dictionary contains all settings of the spline, default to be `None`.
        When it is `None` it will use the default spline settings (5 knots
        determined by the frequency of the data, degree 2, no left and right
        linear tail, no intercept)
    spline_monotonicity
        Spline shape prior, indicate if spline is increasing or
        decreasing.
    spline_convexity
        Spline shape prior, indicate if spline is convex or concave.
    soln_name
        Name of the corresponding covariates multiplier.
    prior_beta_uniform
        Uniform prior for beta, default to None. Otherwise should pass in
        a dictionary with key as the dorm name and value as the uniform prior.
    prior_beta_gaussian
        Same as the `prior_beta_uniform`.

    """

    def __init__(self,
                 cov_name: str,
                 use_spline: bool = False,
                 spline_settings: Optional[Dict] = None,
                 spline_monotonicity: Optional[str] = None,
                 spline_convexity: Optional[str] = None,
                 soln_name: Optional[str] = None,
                 prior_beta_uniform: Optional[Dict] = None,
                 prior_beta_gaussian: Optional[Dict] = None):
        # check the input
        assert isinstance(cov_name, str)
        if spline_monotonicity is not None:
            assert spline_monotonicity in ['increasing', 'decreasing']
        if spline_convexity is not None:
            assert spline_convexity in ['convex', 'concave']
        assert isinstance(soln_name, str) or soln_name is None

        if spline_settings is None:
            spline_settings = {}
        spline_settings = {
            **dict(
                knots=np.linspace(0.0, 1.0, 5),
                degree=2,
                l_linear=True,
                r_linear=True,
                include_first_basis=False,
                knots_type="rel_freq",
            ),
            **spline_settings
        }

        self.cov_name = cov_name
        self.spline = SplineGetter(**spline_settings)
        self.spline_monotonicity = spline_monotonicity
        self.spline_convexity = spline_convexity
        self.use_spline = use_spline
        self.use_constraints = self.use_spline and (
            self.spline_monotonicity is not None or
            self.spline_convexity is not None
        )
        self.soln_name = cov_name if soln_name is None else soln_name
        self.prior_beta_uniform = {} if prior_beta_uniform is None else prior_beta_uniform
        self.prior_beta_gaussian = {} if prior_beta_gaussian is None else prior_beta_gaussian

        if self.use_spline:
            self.num_vars = self.spline.num_spline_bases
        else:
            self.num_vars = 1

    def attach(self, df: DataFrame):
        """Attach the data to variable. It will attach data to the component.
        And create spline and priors if necessary.
        Parameters
        ----------
        df
            The data frame contains the corresponding data column.
        """
        if self.use_spline and isinstance(self.spline, SplineGetter):
            col = df[self.cov_name].to_numpy()
            self.spline = self.spline.get_spline(col)

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
            mat = self.spline.design_mat(cov)
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

    def __repr__(self) -> str:
        return f"CovModel({self.cov_name})"


class CWModel:
    """Cross Walk model.
    """

    def __init__(self,
                 cwdata,
                 obs_type='diff_log',
                 cov_models=None,
                 gold_dorm=None,
                 order_prior=None,
                 use_random_intercept=True,
                 prior_gamma_uniform=None,
                 prior_gamma_gaussian=None):
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
            prior_gamma_uniform (Tuple[float, float], optional):
                If not ``None``, use it as the bound of gamma.
            prior_gamma_gaussian (Tuple[float, float], optional):
                If not ``None``, use it as the gaussian prior of gamma.
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

        # attach data
        for cov_model in self.cov_models:
            cov_model.attach(self.cwdata.df)

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
        self.num_vars_per_dorm = sum([model.num_vars for model in self.cov_models])
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
        self._check_relation_mat()
        self.cov_mat = self.create_cov_mat()
        self._assert_covs_independent()
        self.design_mat = self.create_design_mat()
        self._assert_rank_efficient()
        self.constraint_mat = self.create_constraint_mat()

        # gamma bounds
        self.prior_gamma_uniform = np.array(
            [0.0, np.inf]) if prior_gamma_uniform is None else np.array(prior_gamma_uniform)
        if not self.use_random_intercept:
            self.prior_gamma_uniform = np.zeros(2)
        if self.prior_gamma_uniform[0] < 0.0:
            warnings.warn("Lower bound of gamma has to be non-negative, reset it to zero.")
            self.prior_gamma_uniform[0] = 0.0

        # gamma Gaussian prior
        self.prior_gamma_gaussian = np.array(
            [0.0, np.inf]) if prior_gamma_gaussian is None else np.array(prior_gamma_gaussian)
        if not self.use_random_intercept:
            self.prior_gamma_gaussian = np.array([0.0, np.inf])

        # beta bounds
        uprior = np.repeat(np.array([[-np.inf], [np.inf]]), self.num_vars, axis=1)
        for i, cov_model in enumerate(self.cov_models):
            for dorm, prior in cov_model.prior_beta_uniform.items():
                uprior[:, self.var_idx[dorm][i]] = prior
        uprior[:, self.var_idx[self.gold_dorm]] = 0.0
        self.prior_beta_uniform = uprior

        # beta Gaussian prior
        gprior = np.repeat(np.array([[0.0], [np.inf]]), self.num_vars, axis=1)
        for i, cov_model in enumerate(self.cov_models):
            for dorm, prior in cov_model.prior_beta_gaussian.items():
                gprior[:, self.var_idx[dorm][i]] = prior
        gprior[:, self.var_idx[self.gold_dorm]] = np.array([[0.0], [np.inf]])
        self.prior_beta_gaussian = gprior

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

    def _assert_covs_independent(self):
        """Check if the covariates are independent.
        """
        rank = np.linalg.matrix_rank(self.cov_mat)
        if rank < self.cov_mat.shape[1]:
            raise ValueError("Covariates are collinear, that is, some covariate column is a linear combination of "
                             "some of the other columns. Please check them carefully.")

    def _assert_rank_efficient(self):
        """Check the rank of the design matrix.
        """
        rank = np.linalg.matrix_rank(self.design_mat)
        num_unknowns = self.num_vars_per_dorm*(self.cwdata.num_dorms - 1)
        if rank < num_unknowns:
            raise ValueError(f"Not enough information in the data to recover parameters."
                             f"Number of effective data points is {rank} and number of unknown parameters is {num_unknowns}."
                             f"To reduce the number of parameters, consider converting splines to linear fixed effects "
			      f"or removing covariates entirely.")

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

    def _check_relation_mat(self):
        """Check relation matrix, detect unused dorms.
        """
        col_scales = np.max(np.abs(self.relation_mat), axis=0)
        unused_dorms = [self.cwdata.unique_dorms[i]
                        for i, scale in enumerate(col_scales) if scale == 0.0]
        if len(unused_dorms) != 0:
            raise ValueError(f"{unused_dorms} appears to be unused, most likely it is (they are) "
                             f"in both alt_dorms and ref_dorms at the same time for all its (their) "
                             f"appearance. Please remove {unused_dorms} from alt_dorms and ref_dorms.")

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

    def fit(self,
            max_iter=100,
            inlier_pct=1.0,
            outer_max_iter=100,
            outer_step_size=1.0):
        """Optimize the model parameters.
        This is a interface to limetr.
        Args:
            max_iter (int, optional):
                Maximum number of iterations.
            inlier_pct (float, optional):
                How much percentage of the data do you trust.
            outer_max_iter (int, optional):
                Outer maximum number of iterations.
            outer_step_size (float, optional):
                Step size of the trimming problem, the larger the step size the faster it will converge,
                and the less quality of trimming it will guarantee.
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

        uprior = np.hstack((self.prior_beta_uniform, self.prior_gamma_uniform[:, None]))
        gprior = np.hstack((self.prior_beta_gaussian, self.prior_gamma_gaussian[:, None]))

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
                         gprior=gprior,
                         uprior=uprior,
                         C=cfun,
                         JC=jcfun,
                         c=cvec,
                         inlier_percentage=inlier_pct)
        self.beta, self.gamma, self.w = self.lt.fitModel(inner_print_level=5,
                                                         inner_max_iter=max_iter,
                                                         outer_max_iter=outer_max_iter,
                                                         outer_step_size=outer_step_size)

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
        hessian = self.get_beta_hessian()
        unconstrained_id = np.hstack([
            np.arange(self.lt.k_beta)[self.var_idx[dorm]]
            for dorm in self.cwdata.unique_dorms
            if dorm != self.gold_dorm
        ])
        self.beta_sd = np.zeros(self.lt.k_beta)
        self.beta_sd[unconstrained_id] = np.sqrt(np.diag(
            np.linalg.inv(hessian)
        ))

    def get_beta_hessian(self) -> np.ndarray:
        # compute the posterior distribution of beta
        x = self.lt.JF(self.lt.beta)*np.sqrt(self.lt.w)[:, None]
        z = self.lt.Z*np.sqrt(self.lt.w)[:, None]
        v = limetr.utils.VarMat(self.lt.V**self.lt.w, z, self.lt.gamma, self.lt.n)

        if hasattr(self.lt, 'gprior'):
            beta_gprior_sd = self.lt.gprior[:, self.lt.idx_beta][1]
        else:
            beta_gprior_sd = np.repeat(np.inf, self.lt.k_beta)

        hessian = x.T.dot(v.invDot(x)) + np.diag(1.0/beta_gprior_sd**2)
        hessian = np.delete(hessian, self.var_idx[self.gold_dorm], axis=0)
        hessian = np.delete(hessian, self.var_idx[self.gold_dorm], axis=1)

        return hessian

    def get_cov_names(self) -> List[str]:
        # column of covariate name
        cov_names = []
        for model in self.cov_models:
            if model.spline is None:
                cov_names.append(model.cov_name)
            else:
                cov_names.extend([f'{model.cov_name}_spline_{i}' for i in range(model.num_vars)])
        return cov_names

    def get_fixed_df(self) -> pd.DataFrame:
        """Get fixed effects result data frame.

        Returns:
            pd.DataFrame: Fixed effects result data frame.
        """
        # column of dorms
        dorms = np.repeat(self.cwdata.unique_dorms, self.num_vars_per_dorm)
        cov_names = self.get_cov_names()
        cov_names *= self.cwdata.num_dorms

        # create data frame
        fixed_df = pd.DataFrame({
            'dorm': dorms,
            'name': cov_names,
            'value': self.beta,
            'sd': self.beta_sd,
        })
        return fixed_df

    def get_random_df(self) -> pd.DataFrame:
        """Get random effects result data frame.

        Returns:
            pd.DataFrame: Random effects result data frame.
        """
        gamma_df = pd.DataFrame({
            'name': 'gamma',
            'value': [None] if not hasattr(self, 'lt') else self.lt.gamma
        })
        if self.use_random_intercept:
            int_df = pd.DataFrame({
                'name': self.cwdata.unique_study_id,
                'value': None if not hasattr(self, 'lt') else self.lt.u.ravel()
            })
            return pd.concat([gamma_df, int_df], ignore_index=True)
        return gamma_df

    def get_AIC(self) -> float:
        """Get Akaike information criterion.

        The AIC is equal to 2*k - 2*log(L), where k is the
        number of beta values (fixed-effect coefficients) plus the
        number of gamma values (between-study heterogeneity standard
        deviation), and L is the optimal marginal likelihood value
        where the joint likelihood (fixed and random effects) has been
        integrated with respect to u values (random intercepts).

        Returns:
            float: Akaike information criterion.
        """
        if hasattr(self, 'lt'):
            return 2*(self.lt.k_total + self.lt.objective(self.lt.soln))
        return None

    def summary(self) -> None:
        """Summarize CrossWalk model results.

        Prints fixed effects, random effects, and AIC.
        """
        print('\nFixed effects:\n', self.get_fixed_df())
        print('\nRandom effects:\n', self.get_random_df())
        print('\nAIC:', self.get_AIC())

    def adjust_orig_vals(self, df,
                         orig_dorms,
                         orig_vals_mean,
                         orig_vals_se,
                         study_id=None,
                         data_id=None,
                         ref_dorms=None):
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
            ref_dorms (str, optional):
                Name of the column with reference dorms, if is ``None``, use the
                gold_dorm as the reference dorm. Default to ``None``.

        Returns:
            pandas.DataFrame:
                The adjusted values and standard deviations.
        """
        df_copy = df.copy()
        if ref_dorms is None:
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
                self.random_vars[sid][0]
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
        gamma = np.array([
            self.gamma[0]
            if dorm != self.gold_dorm else 0.0
            for dorm in df[orig_dorms]
        ])

        transformed_ref_vals_mean = transformed_orig_vals_mean - \
            pred_diff_mean - random_effects
        transformed_ref_vals_sd = np.sqrt(transformed_orig_vals_se**2 +
                                          pred_diff_sd**2 + gamma)
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
