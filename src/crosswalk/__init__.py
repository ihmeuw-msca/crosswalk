# -*- coding: utf-8 -*-
"""
crosswalk
~~~~~~~~~

`crosswalk` package.
"""

from threadpoolctl import threadpool_limits

threadpool_limits(limits=1, user_api="blas")
threadpool_limits(limits=1, user_api="openmp")

import os

from . import utils
from .data import *
from .model import *
