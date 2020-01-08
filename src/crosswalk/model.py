# -*- coding: utf-8 -*-
"""
    model
    ~~~~~

    `model` module of the `crosswalk` package.
"""
import numpy as np
from . import data
from . import utils


class CWModel:
    """Cross Walk model.
    """
    def __init__(self, cwdata,
                 obs_type='diff_log',
                 cov_names=None,
                 gold_def=None,
                 order_prior=None,
                 direct_prior=None):
        """Constructor of CWModel.

        Args:
            cwdata (data.CWData):
                Data for cross walk.
            obs_type (str, optional):
                Type of observation can only be chosen from `'diff_log'` and
                `'diff_logit'`.
            cov_names (list{str | list{str}} | None, optional):
                Name of the covarites that will be used in cross walk.
            gold_def (str | None, optional):
                Gold standard definition.
            order_prior (list{list{str}} | None, optional):
                Order priors between different definitions.
            direct_prior (None, optional):
                Direct priors on the covariates multipliers and shared between
                definitions.
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
        self.direct_prior = direct_prior

        self.check()

        # dimensions and indices
        self.num_var_per_def = len(self.cov_names)
        self.num_var = self.num_var_per_def*self.cwdata.num_defs
        indices = utils.sizes_to_indices(np.array([self.num_var_per_def]*
                                                  self.cwdata.num_defs))
        self.var_indices = {
            self.cwdata.unique_defs[i]: indices[i]
            for i in range(self.cwdata.num_defs)
        }
        self.scalar_indices = {
            self.cwdata.unique_defs[i]: i
            for i in range(self.cwdata.num_defs)
        }

        # create the design matrix
        relation_mat = self.create_relation_mat()
        cov_mat = self.create_cov_mat()

        self.design_mat = (
                relation_mat.ravel()[:, None]*
                np.repeat(cov_mat, self.cwdata.num_defs, axis=0)
        ).reshape(self.cwdata.num_obs, self.num_var)

    def check(self):
        """Check the input type, dimension and values.
        """
        assert isinstance(self.cwdata, data.CWData)
        assert self.obs_type in ['diff_log', 'diff_logit'], "Unsupport " \
                                                            "observation type"
        assert isinstance(self.cov_names, list)
        assert self.gold_def in self.cwdata.unique_defs

    def create_relation_mat(self):
        """Creating relation matrix.

        Returns:
            numpy.ndarray:
                Returns relation matrix with 1 encode alternative definition
                and -1 encode reference definition.
        """
        indices = {
            self.cwdata.unique_defs[i]: i
            for i in range(self.cwdata.num_defs)
        }

        relation_mat = np.zeros((self.cwdata.num_obs, self.cwdata.num_defs))
        relation_mat[range(self.cwdata.num_obs), [indices[alt_def]
                         for alt_def in self.cwdata.alt_defs]] = 1.0
        relation_mat[range(self.cwdata.num_obs), [indices[ref_def]
                         for ref_def in self.cwdata.ref_defs]] = -1.0

        return relation_mat

    def create_cov_mat(self):
        """Create covariates matrix.

        Returns:
            numpy.ndarray:
                Returns covarites matrix.
        """
        cov_mat = []
        for cov_name in self.cov_names:
            if isinstance(cov_name, str):
                cov_mat.append(self.cwdata.covs[cov_name][:, None])
            else:
                cov_sub_mat = [self.cwdata.covs[name] for name in cov_name]
                cov_mat.append(np.mean(cov_sub_mat, axis=0)[:, None])

        return np.hstack(cov_mat)
