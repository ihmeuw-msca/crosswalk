# -*- coding: utf-8 -*-
"""
    data
    ~~~~

    `data` module of the `crosswalk` package.
"""
import numpy as np


class CWData:
    """Cross Walk data structure.
    """
    def __init__(self,
                 log_ratio,
                 log_ratio_se,
                 alt_def,
                 ref_def,
                 covs=None,
                 study_id=None,
                 add_intercept=False):
        """Constructor of CWData

        Args:
            log_ratio (numpy.ndarray):
            log_ratio_se (numpy.ndarray):
            alt_def (numpy.ndarray):
            ref_def (numpy.ndarray):
            covs (dict{str: numpy.ndarray} | None, optional):
            study_id (numpy.ndarray | None, optional):
            add_intercept (bool, optional):
        """
