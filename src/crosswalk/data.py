# -*- coding: utf-8 -*-
"""
data
~~~~

`data` module of the `crosswalk` package.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from crosswalk import utils

__all__ = ["CWData"]


class CWData:
    """Cross Walk data structure."""

    def __init__(
        self,
        df: pd.DataFrame,
        obs: str | None = None,
        obs_se: str | None = None,
        alt_dorms: str | None = None,
        ref_dorms: str | None = None,
        dorm_separator: str | None = None,
        covs: list[str] | None = None,
        study_id: str | None = None,
        data_id: str | None = None,
        add_intercept: bool = True,
    ) -> None:
        """Constructor of CWData

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe from csv file that store the data.
        obs : str | None, optional
            Observations of the problem, can be log or logit differences, by default None
        obs_se : str | None, optional
            Standard error of the observations, by default None
        alt_dorms : str | None, optional
            Alternative definitions/methods for each observation, by default None
        ref_dorms : str | None, optional
            Reference definitions/methods for each observation, by default None
        dorm_separator : str | None, optional
            Used when there are multiple definitions in alt_dorms or ref_dorms.
            Will decompose the dorm by this separator into multiple dorms.
            If None, assume single dorm for alt_dorms and ref_dorms, by default None
        covs : list[str], optional
            Covariates linearly parametrized the observation, by default None
        study_id : str | None, optional
            Study id for each observation, by default None
        data_id : str | None, optional
            ID for each data, if pass in column, it requires the elements in the column
            to be different from each other. If ``None``, the program will generate
            a integer sequence from 0 to ``num_obs`` to serve as the data_id, by default None
        add_intercept : bool, optional
            If `True`, add intercept to the current covariates, by default True
        """
        self.df = df

        self.col_obs = obs
        self.col_obs_se = obs_se
        self.col_alt_dorms = alt_dorms
        self.col_ref_dorms = ref_dorms
        self.col_covs = covs
        self.col_study_id = study_id
        self.col_data_id = data_id

        self.obs = None if self.col_obs is None else df[self.col_obs].values
        self.obs_se = None if obs_se is None else df[obs_se].values
        self.dorm_separator = dorm_separator
        alt_dorms = (
            alt_dorms
            if self.col_alt_dorms is None
            else df[self.col_alt_dorms].to_numpy().astype(str)
        )
        ref_dorms = (
            ref_dorms
            if self.col_ref_dorms is None
            else df[self.col_ref_dorms].to_numpy().astype(str)
        )
        self.alt_dorms = utils.process_dorms(
            dorms=alt_dorms,
            size=self.df.shape[0],
            default_dorm="alt",
            dorm_separator=self.dorm_separator,
        )
        self.ref_dorms = utils.process_dorms(
            dorms=ref_dorms,
            size=self.df.shape[0],
            default_dorm="ref",
            dorm_separator=self.dorm_separator,
        )

        self.covs = (
            pd.DataFrame() if self.col_covs is None else df[self.col_covs].copy()
        )
        self.study_id = (
            None if self.col_study_id is None else df[self.col_study_id].values
        )
        self.data_id = (
            np.arange(self.df.shape[0])
            if self.col_data_id is None
            else df[self.col_data_id].values
        )

        # dimensions of observations and covariates
        self.num_obs = self.df.shape[0]
        if self.covs.empty and not add_intercept:
            warnings.warn(
                "Covariates must at least include intercept."
                "Adding intercept automatically."
            )
            add_intercept = True

        if add_intercept:
            self.covs["intercept"] = np.ones(self.num_obs)

        self.num_covs = self.covs.shape[1]

        # check inputs
        self.check()

        # definition structure
        self.num_dorms, self.dorm_sizes, self.unique_dorms = utils.array_structure(
            self.alt_dorms + self.ref_dorms
        )
        self.num_alt_dorms, self.alt_dorm_sizes, self.unique_alt_dorms = (
            utils.array_structure(self.alt_dorms)
        )
        self.num_ref_dorms, self.ref_dorm_sizes, self.unique_ref_dorms = (
            utils.array_structure(self.ref_dorms)
        )
        self.max_dorm = self.unique_dorms[np.argmax(self.dorm_sizes)]
        self.min_dorm = self.unique_dorms[np.argmin(self.dorm_sizes)]
        self.max_alt_dorm = self.unique_alt_dorms[np.argmax(self.alt_dorm_sizes)]
        self.min_alt_dorm = self.unique_alt_dorms[np.argmin(self.alt_dorm_sizes)]
        self.max_ref_dorm = self.unique_ref_dorms[np.argmax(self.ref_dorm_sizes)]
        self.min_ref_dorm = self.unique_ref_dorms[np.argmin(self.ref_dorm_sizes)]

        self.dorm_idx = {dorm: i for i, dorm in enumerate(self.unique_dorms)}

        # study structure
        if self.study_id is None:
            self.num_studies = 0
            self.study_sizes = np.array([])
            self.unique_study_id = None
        else:
            self.num_studies, self.study_sizes, self.unique_study_id = (
                utils.array_structure(self.study_id)
            )
        self.sort_by_study_id()

    def check(self) -> None:
        """Check inputs type, shape and value."""
        # obs_se validation
        if self.obs is not None and not utils.is_numerical_array(
            self.obs, shape=(self.num_obs,)
        ):
            raise ValueError(
                f"Expected 'obs' to be None or a numerical array of shape ({self.num_obs},), "
                f"but received invalid input of type {type(self.obs).__name__}."
            )
        if self.obs_se is not None and not utils.is_numerical_array(
            self.obs_se, shape=(self.num_obs,)
        ):
            raise ValueError(
                f"Expected 'obs_se' to be None or a numerical array of shape ({self.num_obs},)."
            )

        if utils.is_numerical_array(self.obs_se) and not (self.obs_se > 0.0).all():
            raise ValueError(
                "All values in 'obs_se' must be strictly greater than 0.0."
            )

        # alt_dorms and ref_dorms validation
        if not isinstance(self.alt_dorms, list):
            raise TypeError(
                f"Expected 'alt_dorms' to be a list, got {type(self.alt_dorms).__name__}."
            )
        if not isinstance(self.ref_dorms, list):
            raise TypeError(
                f"Expected 'ref_dorms' to be a list, got {type(self.ref_dorms).__name__}."
            )
        if len(self.alt_dorms) != self.num_obs:
            raise ValueError(
                f"Expected 'alt_dorms' to have length {self.num_obs}, got {len(self.alt_dorms)}."
            )
        if len(self.ref_dorms) != self.num_obs:
            raise ValueError(
                f"Expected 'ref_dorms' to have length {self.num_obs}, got {len(self.ref_dorms)}."
            )

        # covs validation
        if not isinstance(self.covs, pd.DataFrame):
            raise TypeError(
                f"Expected 'covs' to be a pandas DataFrame, got {type(self.covs).__name__}."
            )
        if self.covs.shape[1] != self.num_covs:
            raise ValueError(
                f"Expected 'covs' to have {self.num_covs} columns, got {self.covs.shape[1]}."
            )

        if self.study_id is not None and self.study_id.shape != (self.num_obs,):
            raise ValueError(
                f"Expected 'study_id' to have shape ({self.num_obs},), got {self.study_id.shape}."
            )
        if len(set(self.data_id)) != self.num_obs:
            raise ValueError("data_id has to be unique for each data point.")

    def sort_by_study_id(self) -> None:
        """Sort the observations and covariates by the study id."""
        if self.study_id is not None:
            sort_id = np.argsort(self.study_id)
            self.study_id = self.study_id[sort_id]
            self.data_id = self.data_id[sort_id]
            self.obs = self.obs[sort_id]
            self.obs_se = self.obs_se[sort_id]
            self.alt_dorms = [self.alt_dorms[index] for index in sort_id]
            self.ref_dorms = [self.ref_dorms[index] for index in sort_id]
            self.covs = self.covs.reindex(sort_id)

    def copy_dorm_structure(self, cwdata: CWData) -> None:
        """Copy the dorm structure from other"""
        if cwdata.num_dorms < self.num_dorms:
            raise ValueError(
                "number of dorms in provided must be greater than or equal to number of dorms in current CWData instance"
            )
        if not set(self.unique_dorms).issubset(cwdata.unique_dorms):
            raise ValueError(
                "at least 1 dorm in this instance's unique_dorms is not in the provided CWData's unique dorms"
            )

        self.num_dorms = cwdata.num_dorms
        self.unique_dorms = cwdata.unique_dorms
        self.dorm_idx = cwdata.dorm_idx

    def __repr__(self) -> str:
        """Summary of the object."""
        dimension_summary = [
            f"number of observations: {self.num_obs}",
            f"number of covariates  : {self.num_covs}",
            f"number of defs/methods: {self.num_dorms}",
            f"number of studies     : {self.num_studies}",
        ]
        return "\n".join(dimension_summary)
