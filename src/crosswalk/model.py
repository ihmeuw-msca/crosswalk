# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
import numpy as np
from limetr import LimeTr
from . import data
from . import utils


class CWModel:
    """Cross Walk model.
    """
    def __init__(self, cwdata,
                 obs_type='diff_log',
                 cov_names=None,
                 gold_def=None,
                 order_prior=None):
        """Constructor of CWModel.

        Args:
            cwdata (data.CWData):
                Data for cross walk.
            obs_type (str, optional):
                Type of observation can only be chosen from `'diff_log'` and
                `'diff_logit'`.
            cov_names (list{str} | None, optional):
                Name of the covarites that will be used in cross walk.
            gold_def (str | None, optional):
                Gold standard definition.
            order_prior (list{list{str}} | None, optional):
                Order priors between different definitions.
        """
        self.cwdata = cwdata
        self.obs_type = obs_type
        if cov_names is None:
            self.cov_names = list(self.cwdata.covs.keys())
        else:
            self.cov_names = cov_names

        if gold_def is None:
            unique_ref_defs, ref_defs_counts = np.unique(self.cwdata.ref_defs,
                                                         return_counts=True)
            self.gold_def = unique_ref_defs[np.argmax(ref_defs_counts)]
        else:
            self.gold_def = gold_def

        self.order_prior = order_prior

        self.check()

        # dimensions and indices
        self.num_var_per_def = len(self.cov_names)
        self.num_var = self.num_var_per_def*self.cwdata.num_defs
        indices = utils.sizes_to_indices(np.array([self.num_var_per_def] *
                                                  self.cwdata.num_defs))
        self.var_indices = {
            self.cwdata.unique_defs[i]: indices[i]
            for i in range(self.cwdata.num_defs)
        }
        self.def_indices = {
            self.cwdata.unique_defs[i]: i
            for i in range(self.cwdata.num_defs)
        }

        # create the design and constraint matrix
        self.design_mat = self.create_design_mat()
        self.constraint_mat = self.create_constraint_mat()

        # place holder for the result
        self.beta = None
        self.gamma = None

    def check(self):
        """Check the input type, dimension and values.
        """
        assert isinstance(self.cwdata, data.CWData)
        assert self.obs_type in ['diff_log', 'diff_logit'], "Unsupport " \
                                                            "observation type"
        assert isinstance(self.cov_names, list)
        assert self.gold_def in self.cwdata.unique_defs

    @property
    def relation_mat(self):
        """Creating relation matrix.

        Returns:
            numpy.ndarray:
                Returns relation matrix with 1 encode alternative definition
                and -1 encode reference definition.
        """

        relation_mat = np.zeros((self.cwdata.num_obs, self.cwdata.num_defs))
        relation_mat[range(self.cwdata.num_obs), [self.def_indices[alt_def]
                     for alt_def in self.cwdata.alt_defs]] = 1.0
        relation_mat[range(self.cwdata.num_obs), [self.def_indices[ref_def]
                     for ref_def in self.cwdata.ref_defs]] = -1.0

        return relation_mat

    @property
    def cov_mat(self):
        """Create covariates matrix.

        Returns:
            numpy.ndarray:
                Returns covarites matrix.
        """
        return np.hstack([self.cwdata.covs[cov_name][:, None]
                          for cov_name in self.cov_names])

    def create_design_mat(self):
        """Create linear design matrix.

        Returns:
            numpy.ndarray:
                Returns linear design matrix.
        """

        design_mat = (
                self.relation_mat.ravel()[:, None] *
                np.repeat(self.cov_mat, self.cwdata.num_defs, axis=0)
        ).reshape(self.cwdata.num_obs, self.num_var)

        return design_mat

    def create_constraint_mat(self):
        """Create constraint matrix.

        Returns:
            numpy.ndarray:
                Return constraints matrix.
        """
        if self.order_prior is None:
            return None

        mat = []
        for i, d in enumerate(self.order_prior):
            sub_mat = np.zeros((self.num_var_per_def, self.num_var))
            sub_mat[:, self.var_indices[d[0]]] = -np.eye( self.num_var_per_def)
            sub_mat[:, self.var_indices[d[1]]] = np.eye(self.num_var_per_def)
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
        k_beta = self.num_var
        k_gamma = 1
        Y = self.cwdata.obs
        S = self.cwdata.obs_se
        X = self.design_mat
        Z = np.ones((self.cwdata.num_obs, 1))

        uprior = np.array([[-np.inf]*self.num_var + [0.0],
                           [np.inf]*self.num_var + [np.inf]])
        uprior[:, self.var_indices[self.gold_def]] = 0.0

        def F(beta):
            return X.dot(beta)

        def JF(beta):
            return X

        if self.constraint_mat is not None:
            num_constraints = self.constraint_mat.shape[0]
            C_mat = np.hstack((self.constraint_mat,
                               np.zeros((num_constraints, 1))))
            c = np.array([[0.0]*num_constraints,
                          [np.inf]*num_constraints])

            def C(beta):
                return C_mat.dot(beta)

            def JC(beta):
                return C_mat

        else:
            C = None
            JC = None
            c = None

        lt = LimeTr(n, k_beta, k_gamma, Y, F, JF, Z, S=S,
                    uprior=uprior, C=C, JC=JC, c=c)
        beta, gamma, _ = lt.fitModel(inner_print_level=5,
                                     inner_max_iter=max_iter)

        self.beta = {
            d: beta[self.var_indices[d]]
            for d in self.cwdata.unique_defs
        }
        self.gamma = gamma
