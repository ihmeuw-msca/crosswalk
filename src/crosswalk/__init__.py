# -*- coding: utf-8 -*-
"""
crosswalk
~~~~~~~~~

`crosswalk` package.
"""

from threadpoolctl import threadpool_limits

from crosswalk.data import CWData
from crosswalk.model import CovModel, CWModel

threadpool_limits(limits=1, user_api="blas")
threadpool_limits(limits=1, user_api="openmp")

__all__ = ["CWData", "CovModel", "CWModel", "utils"]
