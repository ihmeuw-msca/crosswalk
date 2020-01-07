# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module of the `crosswalk` package.
"""
import numpy as np
import warnings
from . import utils


class CWData:
    """Cross Walk data structure.
    """
    def __init__(self,
                 log_ratio,
                 log_ratio_se,
                 alt_defs,
                 ref_defs,
                 covs=None,
                 study_id=None,
                 add_intercept=False):
        """Constructor of CWData

        Args:
            log_ratio (numpy.ndarray):
            log_ratio_se (numpy.ndarray):
            alt_defs (numpy.ndarray):
            ref_defs (numpy.ndarray):
            covs (dict{str: numpy.ndarray} | None, optional):
            study_id (numpy.ndarray | None, optional):
            add_intercept (bool, optional):
        """
        self.log_ratio = log_ratio
        self.log_ratio_se = log_ratio_se
        self.alt_defs = alt_defs
        self.ref_defs = ref_defs
        self.covs = {} if covs is None else covs
        self.study_id = study_id

        # dimensions of observations and covariates
        self.num_obs = self.log_ratio.size
        if not self.covs and not add_intercept:
            warnings.warn("Covariates must at least include intercept."
                          "Adding intercept automatically.")
            add_intercept = True

        if add_intercept:
            self.covs.update({'intercept': np.ones(self.num_obs)})

        self.num_covs = len(self.covs)

        # check inputs
        self.check()

        # definition structure
        self.num_defs, self.unique_defs = self.def_structure()

        # study structure
        (self.num_studies,
         self.study_sizes,
         self.unique_study_id) = self.study_structure()
        self.sort_by_study_id()


    def check(self):
        """Check inputs type, shape and value.
        """
        assert utils.is_numerical_array(self.log_ratio,
                                        shape=(self.num_obs,))
        assert utils.is_numerical_array(self.log_ratio_se,
                                        shape=(self.num_obs,))
        assert (self.log_ratio_se > 0.0).all()

        assert isinstance(self.alt_defs, np.ndarray)
        assert isinstance(self.ref_defs, np.ndarray)
        assert self.alt_defs.shape == (self.num_obs,)
        assert self.ref_defs.shape == (self.num_obs,)

        assert isinstance(self.covs, dict)
        assert len(self.covs) == self.num_covs
        for cov_name in self.covs:
            assert isinstance(cov_name, str)
            assert utils.is_numerical_array(self.covs[cov_name],
                                            shape=(self.num_obs,))

        if self.study_id is not None:
            assert utils.is_numerical_array(self.study_id,
                                            shape=(self.num_obs,))

    def study_structure(self):
        """Obtain study structure from study_id

        Returns:
            tuple{int, numpy.ndarray, numpy.ndarray}:
                Return the number of studies, each study size and unique study
                id.
        """
        if self.study_id is None:
            num_studies = self.num_obs
            study_sizes = np.array([1]*self.num_obs)
            unique_study_id = None
        else:
            unique_study_id, study_sizes = np.unique(self.study_id,
                                                     return_counts=True)
            num_studies = unique_study_id.size

        return num_studies, study_sizes, unique_study_id

    def def_structure(self):
        """Obtain definition structure from alt_defs and ref_defs

        Returns:
            tuple{int, numpy.ndarray}:
                Return the number of the definitions and unique definitions.
        """
        unique_defs = np.unique(np.hstack((self.alt_defs, self.ref_defs)))
        num_defs = unique_defs.size

        return num_defs, unique_defs

    def sort_by_study_id(self):
        """Sort the observations and covariates by the study id.
        """
        if self.study_id is not None:
            sort_id = np.argsort(self.study_id)
            self.study_id = self.study_id[sort_id]
            self.log_ratio = self.log_ratio[sort_id]
            self.log_ratio_se = self.log_ratio_se[sort_id]
            self.alt_defs = self.alt_defs[sort_id]
            self.ref_defs = self.ref_defs[sort_id]
            for cov_name in self.covs:
                self.covs[cov_name] = self.covs[cov_name][sort_id]


    def __repr__(self):
        """Summary of the object.
        """
        dimension_summary = [
            "number of observations: %i" % self.num_obs,
            "number of covariates  : %i" % self.num_covs,
            "number of definitions : %i" % self.num_defs,
            "number of studies     : %i" % self.num_studies,
        ]
        return "\n".join(dimension_summary)
