# -*- coding: utf-8 -*-
"""
crosswalk
~~~~~~~~~

`crosswalk` package.
"""

from threadpoolctl import threadpool_limits

from crosswalk import utils
from crosswalk.data import CWData
from crosswalk.model import CovModel, CWModel
from crosswalk.plots import dose_response_curve, funnel_plot
from crosswalk.post_analysis import PostAnalysis
from crosswalk.scorelator import Scorelator

threadpool_limits(limits=1, user_api="blas")
threadpool_limits(limits=1, user_api="openmp")

__all__ = [
    "utils",
    "CWData",
    "CovModel",
    "CWModel",
    "dose_response_curve",
    "funnel_plot",
    "PostAnalysis",
    "Scorelator",
]
