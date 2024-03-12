import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from crosswalk.data import CWData
from crosswalk.model import CWModel
from crosswalk import utils


def _check_cov_alignment(
    cwmodel: CWModel,
    dose_variable: str,
    continuous_variables: list[str],
    binary_variables: dict[str, float],
) -> None:
    """Check if all non-dose and non-intercept variables are provided with
    reference values. And if there are any extra variables that are not in the model.

    Parameters
    ----------
    cwmodel : CWModel
        Fitted CrossWalk model object.
    dose_variable : str
        Dose variable name.
    continuous_variables : list
        List of continuous covariate names.
    binary_variables : dict
        A dictionary to specify the values for binary variables.
        Options for values: 'median', 'mean', or certain value
        Example: binary

    Raises
    ------
    ValueError
        If not all non-dose variables are provided with reference values.
        If there are any extra variables that are not in the model.

    TODO
    ----
    This can be done in a smarter way
    - Does not distinguish between continuous and binary variables, we should allow
      the user pass in the reference values for variables in one dictionary.
    - We should define default behavior to avoid this error checking, if not specified
      default to use median. If specify extra, remove it from the list. However
      this will make the function less transparent, it is up-to-debate.

    """
    cwmodel_covs = [cov_model.cov_name for cov_model in cwmodel.cov_models]
    specified_covs = (
        [dose_variable] + continuous_variables + list(binary_variables.keys())
    )
    if "intercept" in cwmodel_covs:
        specified_covs.append("intercept")
    if set(cwmodel_covs) != set(specified_covs):
        missing_covs = set(cwmodel_covs) - set(specified_covs)
        extra_covs = set(specified_covs) - set(cwmodel_covs)
        raise ValueError(
            "Must provide reference values for all variables in the model, except"
            "for does variable and intercept."
            "And must not specify any extra variables that are not in the model."
            f"Current missing covariates: {missing_covs}."
            f"Current extra covariates: {extra_covs}."
        )


def _get_point_data(
    dose_variable: str, cwdata: CWData, cwmodel: CWModel
) -> pd.DataFrame:
    df = cwdata.df.reset_index(drop=True)
    data = pd.DataFrame(
        {
            "y": df[cwdata.col_obs],
            "se": df[cwdata.col_obs_se],
            "w": cwmodel.lt.w,
            dose_variable: df[dose_variable],
            "obs_method": df[cwdata.col_alt_dorms],
            "dorm_alt": df[cwdata.col_alt_dorms],
            "dorm_ref": df[cwdata.col_ref_dorms],
            "intercept": 1.0,
            # bolier variables to use adjust_orig_vals function
            "orig_vals_mean": 0.1,
            "orig_vals_se": 0.1,
        }
    )

    data = pd.concat(
        [data, df[[cov for cov in cwdata.col_covs if cov not in data]]], axis=1
    )

    data["y_pred"] = cwmodel.adjust_orig_vals(
        df=data,
        orig_dorms="obs_method",
        orig_vals_mean="orig_vals_mean",
        orig_vals_se="orig_vals_se",
    )["pred_diff_mean"]

    # determine points inside/outside funnel
    data["position"] = "inside funnel"
    data.loc[data.eval("abs(y - y_pred) > 1.96 * se"), "position"] = "outside funnel"

    # get inlier/outlier
    data["trim"] = "inlier"
    data.loc[data.eval("w < 0.6"), "trim"] = "outlier"

    # get plot guide
    data["plot_guide"] = data["trim"] + ", " + data["position"]

    # get scaled marker size
    data["size_var"] = data.eval("1.0 / se")
    data["size_var"] = data.eval("300 * size_var / size_var.max()")

    return data


def _get_curve_data(
    dose_variable: str,
    obs_method: str,
    continuous_variables: list[str],
    binary_variables: dict[str, float],
    cwdata: CWData,
    cwmodel: CWModel,
    from_zero: bool,
    include_bias: bool,
) -> pd.DataFrame:
    df = cwdata.df.copy()
    # check for knots
    min_cov = 0.0 if from_zero else df[dose_variable].min()
    max_cov = cwdata.df[dose_variable].max()

    # construct dataframe for prediction
    cov_range = max_cov - min_cov
    dose_grid = np.arange(min_cov, max_cov + cov_range * 0.001, cov_range / 1000)

    data = pd.DataFrame(
        {
            dose_variable: dose_grid,
            "obs_method": obs_method,
            "intercept": 1.0,
            # bolier variables to use adjust_orig_vals function
            "orig_vals_mean": 0.1,
            "orig_vals_se": 0.1,
        }
    )

    if include_bias:
        for cov in continuous_variables:
            data[cov] = df[cov].median()
        for cov, value in binary_variables.items():
            if value in ["mean", "median"]:
                data[cov] = df.eval(f"{cov}.{value}()")
            else:
                data[cov] = value
    else:
        for cov in continuous_variables + list(binary_variables.keys()):
            data[cov] = 0.0

    pred = cwmodel.adjust_orig_vals(
        df=data,
        orig_dorms="obs_method",
        orig_vals_mean="orig_vals_mean",
        orig_vals_se="orig_vals_se",
    )

    data["y_mean"] = pred["pred_diff_mean"]
    data["y_sd_fe"] = pred["pred_diff_sd"]
    data["y_lo_fe"] = data.eval("y_mean - 1.96 * y_sd_fe")
    data["y_hi_fe"] = data.eval("y_mean + 1.96 * y_sd_fe")
    data["y_sd"] = data.eval(f"sqrt(y_sd_fe ** 2 + {cwmodel.gamma[0]})")
    data["y_lo"] = data.eval("y_mean - 1.96 * y_sd")
    data["y_hi"] = data.eval("y_mean + 1.96 * y_sd")

    return data


def _plot_dose_response_curve(
    dose_variable: str,
    obs_method: str,
    gold_dorm: str,
    point_data: pd.DataFrame,
    curve_data: pd.DataFrame,
    title: str,
    plot_note: str | None = None,
    knots: list[float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    # plot
    sns.set_style("whitegrid")
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 0.5
    fontsize = dict(body=10, title=12)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(labelsize=fontsize["body"])

    plot_key = {
        "inlier, inside funnel": dict(
            marker="o", facecolors="seagreen", edgecolors="darkgreen"
        ),
        "inlier, outside funnel": dict(
            marker="o", facecolors="coral", edgecolors="firebrick"
        ),
        "outlier, inside funnel": dict(marker="x", facecolors="darkgreen"),
        "outlier, outside funnel": dict(marker="x", facecolors="firebrick"),
    }

    ax.fill_between(
        curve_data[dose_variable],
        curve_data["y_lo"],
        curve_data["y_hi"],
        alpha=0.5,
        color="lightgrey",
    )
    ax.fill_between(
        curve_data[dose_variable],
        curve_data["y_lo_fe"],
        curve_data["y_hi_fe"],
        alpha=0.75,
        color="darkgrey",
    )
    ax.plot(
        curve_data[dose_variable], curve_data["y_mean"], color="black", linewidth=0.75
    )
    # plt.xlim([min_cov, max_cov])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel("Exposure", fontsize=fontsize["body"])
    ax.set_ylabel("Effect size", fontsize=fontsize["body"])

    # other comparison
    non_direct_df = point_data.query(
        f"(dorm_ref != '{gold_dorm}') | (dorm_alt != '{obs_method}')"
    )
    # direct comparison
    plot_data_df = point_data.query(
        f"(dorm_ref == '{gold_dorm}') & (dorm_alt == '{obs_method}')"
    )

    for key, value in plot_key.items():
        df_sub = plot_data_df.query(f"plot_guide == '{key}'")
        ax.scatter(
            df_sub[dose_variable],
            df_sub["y"],
            s=df_sub["size_var"],
            linewidth=0.6,
            alpha=0.6,
            label=key,
            **value,
        )

    if not non_direct_df.empty:
        ax.scatter(
            non_direct_df[dose_variable],
            non_direct_df["y"],
            facecolors="grey",
            edgecolors="grey",
            alpha=0.3,
            label="Other comparison",
        )

    # Plot title
    ax.set_title(title, fontsize=10)
    if plot_note is not None:
        fig.suptitle(plot_note, y=1.01, fontsize=fontsize["title"])

    ax.legend(loc="upper left")

    if knots is not None:
        for x in knots:
            plt.axvline(x, color="navy", linestyle="--", alpha=0.5, linewidth=0.75)

    return fig


def _get_title(obs_method: str, cwmodel: CWModel) -> str:
    var_sizes = [cov_model.num_vars for cov_model in cwmodel.cov_models]
    cov_names = [cov_model.cov_name for cov_model in cwmodel.cov_models]

    beta = cwmodel.fixed_vars[obs_method]
    beta_dict = dict(zip(cov_names, np.split(beta, np.cumsum(var_sizes[:-1]))))

    for key, value in beta_dict.items():
        value = list(value)
        if len(value) == 1:
            beta_dict[key] = value[0]

    title = "; ".join([f"{key}: {value}" for key, value in beta_dict.items()])
    return title


def _get_knots(dose_variable: str, cwmodel: CWModel) -> list[float] | None:
    cov_names = [cov_model.cov_name for cov_model in cwmodel.cov_models]
    cov_model = cwmodel.cov_models[cov_names.index(dose_variable)]
    if cov_model.spline is None:
        return None
    return list(cov_model.spline.knots)


def dose_response_curve(
    dose_variable: str,
    obs_method: str,
    continuous_variables: list[str] | None = None,
    binary_variables: dict[str, float] | None = None,
    plots_dir: str | None = None,
    cwdata: CWData | None = None,
    cwmodel: CWModel | None = None,
    file_name: str = "dose_response_plot",
    from_zero: bool = False,
    include_bias: bool = False,
    ylim: tuple[float, float] | None = None,
    plot_note: str | None = None,
    write_file: bool = False,
):
    """Plot dose response curve. Crosswalk model with spline on dose variable
    to parametrize the difference between the reference and alternative definitions.

    Parameters
    ----------
    dose_variable : str
        Dose variable name.
    obs_method : str
        Alternative definition or method intended to be plotted.
    continuous_variables : list, optional
        List of continuous covariate names.
    binary_variables : dict, optional
        A dictionary to specify the values for binary variables.
        Options for values: 'median', 'mean', or certain value
        Example: binary_variables = {'sex_id': 1, 'age_id': 'median'}
    plots_dir : str, optional
        Directory where to save the plot.
    cwdata : CWData, optional
        CrossWalk data object.
    cwmodel : CWModel, optional
        Fitted CrossWalk model object.
    from_zero : bool, optional
        If set to be True, y-axis will start from zero.
    ylim : tuple, optional
        y-axis bound. E.g. [0, 10]
    file_name : str, optional
        File name for the plot.
    plot_note : str, optional
        The notes intended to be written on the title.
    include_bias : bool, optional
        Whether to include bias or not.
    write_file : bool, optional
        Specify `True` if the plot is expected to be saved on disk.
        If True, `plots_dir` should be specified too.

    """
    # process input
    continuous_variables = continuous_variables or []
    binary_variables = binary_variables or {}
    continuous_variables = list(set(continuous_variables) - set([dose_variable]))

    # All covariates in cwmodel should be specified.
    _check_cov_alignment(cwmodel, dose_variable, continuous_variables, binary_variables)

    # Extract data for plotting data points
    point_data = _get_point_data(dose_variable, cwdata, cwmodel)

    # Extract data for plotting curves
    curve_data = _get_curve_data(
        dose_variable,
        obs_method,
        continuous_variables,
        binary_variables,
        cwdata,
        cwmodel,
        from_zero,
        include_bias,
    )

    # Plot dose response curve
    title = _get_title(obs_method, cwmodel)
    knots = _get_knots(dose_variable, cwmodel)
    fig = _plot_dose_response_curve(
        dose_variable,
        obs_method,
        cwmodel.gold_dorm,
        point_data,
        curve_data,
        title,
        plot_note,
        knots,
        ylim,
    )

    # Save plots
    if write_file:
        assert plots_dir is not None, "plots_dir is not specified!"
        outfile = Path(plots_dir) / f"{file_name}.pdf"
        fig.savefig(outfile, orientation="landscape", bbox_inches="tight")
        print(f"Dose response plot saved at {outfile}")
    else:
        fig.show()
    plt.close()


def funnel_plot(
    obs_method="Self-reported",
    cwdata=None,
    cwmodel=None,
    continuous_variables=[],
    binary_variables={},
    plots_dir=None,
    file_name="funnel_plot",
    plot_note=None,
    include_bias=False,
    write_file=False,
):
    """Funnel Plot.
    Args:
        obs_method (str):
            Alternative definition or method intended to be plotted.
        cwdata (CWData object):
            CrossWalk data object.
        cwmodel (CWModel object):
            Fitted CrossWalk model object.
        continuous_variables (list):
            List of continuous covariate names.
        binary_variables (dict):
            A dictionary to specify the values for binary variables.
            Options for values: 'median', 'mean', or certain value
            Example: binary_variables = {'sex_id': 1, 'age_id': 'median'}
        plots_dir (str):
            Directory where to save the plot.
        file_name (str):
            File name for the plot.
        plot_note (str):
            The notes intended to be written on the title.
        include_bias (bool):
            Whether to include bias or not.
        write_file (bool):
            Specify `True` if the plot is expected to be saved on disk.
            If True, `plots_dir` should be specified too.

    """
    assert obs_method in cwdata.unique_alt_dorms, f"{obs_method} not in alt_dorms!"

    data_df = pd.DataFrame(
        {
            "y": cwdata.obs,
            "se": cwdata.obs_se,
            "w": cwmodel.lt.w,
            "dorm_alt": cwdata.df[cwdata.col_alt_dorms].values,
            "dorm_ref": cwdata.df[cwdata.col_ref_dorms].values,
        }
    )

    # determine points inside/outside funnel
    data_df["position"] = "other comparison"
    data_df.loc[
        (data_df.dorm_ref == cwmodel.gold_dorm) & (data_df.dorm_alt == obs_method),
        "position",
    ] = "direct comparison"

    # get inlier/outlier
    data_df.loc[data_df.w >= 0.6, "trim"] = "inlier"
    data_df.loc[data_df.w < 0.6, "trim"] = "outlier"

    # get plot guide
    data_df["plot_guide"] = data_df["trim"] + ", " + data_df["position"]
    plot_key = {
        "inlier, other comparison": ("o", "grey", "grey", 0.3),
        "inlier, direct comparison": ("o", "coral", "firebrick", 0.75),
        "outlier, other comparison": ("x", "grey", "grey", 0.3),
        "outlier, direct comparison": ("x", "firebrick", "firebrick", 0.75),
    }

    # construct dataframe for prediction, prev and prev_se don't matter.
    pred_df = pd.DataFrame(
        {"obs_method": obs_method, "prev": 0.1, "prev_se": 0.1}, index=[0]
    )
    # if it's continuous variable, take median
    for var in continuous_variables:
        pred_df[var] = np.median(cwdata.covs[var])

    # if binary_variables specified, process binary variables accordingly
    if binary_variables:
        for var in binary_variables.keys():
            value = binary_variables.get(var)
            if value == "mean":
                pred_df[var] = np.mean(data_df[var])
            elif value == "median":
                pred_df[var] = np.median(data_df[var])
            else:
                pred_df[var] = value

    # predict effect
    y_pred = cwmodel.adjust_orig_vals(
        df=pred_df,
        orig_dorms="obs_method",
        orig_vals_mean="prev",
        orig_vals_se="prev_se",
    ).to_numpy()
    y_pred = np.ravel(y_pred)
    y_mean, y_sd = y_pred[2], y_pred[3]
    # Include random effects
    y_sd = np.sqrt(y_sd**2 + cwmodel.gamma[0])

    # Statistics in title
    y_lower, y_upper = y_mean - 1.96 * y_sd, y_mean + 1.96 * y_sd
    p_value = utils.p_value(np.array([y_mean]), np.array([y_sd]))[0]
    content_string = f"Mean effect: {np.round(y_mean, 3)}\
    (95% CI: {np.round(y_lower, 3)} to {np.round(y_upper, 3)});\
    p-value: {np.round(p_value, 4)}"

    # triangle
    max_se = cwdata.obs_se.max()
    se_domain = np.arange(0, max_se * 1.1, max_se / 100)
    se_lower = y_mean - (se_domain * 1.96)
    se_upper = y_mean + (se_domain * 1.96)

    sns.set_style("darkgrid")
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 0.5
    plt.figure(figsize=(10, 8))
    plt.fill_betweenx(se_domain, se_lower, se_upper, color="white", alpha=0.75)
    plt.axvline(
        y_mean,
        0,
        1 - (0.025 * max(se_domain) / (max(se_domain) * 1.025)),
        color="black",
        alpha=0.75,
        linewidth=0.75,
    )
    plt.plot(se_lower, se_domain, color="black", linestyle="--", linewidth=0.75)
    plt.plot(se_upper, se_domain, color="black", linestyle="--", linewidth=0.75)
    plt.ylim([-0.025 * max(se_domain), max(se_domain)])
    plt.xlabel("Effect size", fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel("Standard error", fontsize=10)
    plt.yticks(fontsize=10)
    plt.axvline(0, color="mediumseagreen", alpha=0.75, linewidth=0.75)
    # Plot inlier and outlier
    for key, value in plot_key.items():
        plt.plot(
            data_df.loc[data_df.plot_guide == key, "y"],
            data_df.loc[data_df.plot_guide == key, "se"],
            "o",
            markersize=5,
            marker=value[0],
            markerfacecolor=value[1],
            markeredgecolor=value[2],
            markeredgewidth=0.6,
            alpha=value[3],
            label=key,
        )

    plt.legend(loc="upper left", frameon=False)
    plt.gca().invert_yaxis()
    # Plot title
    if plot_note is not None:
        plt.title(content_string, fontsize=10)
        plt.suptitle(plot_note, y=1.01, fontsize=12)
    else:
        plt.title(content_string, fontsize=10)
    # Save plots
    if write_file:
        assert plots_dir is not None, "plots_dir is not specified!"
        outfile = os.path.join(plots_dir, file_name + ".pdf")
        plt.savefig(outfile, orientation="landscape", bbox_inches="tight")
        print(f"Funnel plot saved at {outfile}")
    else:
        plt.show()
    plt.close()
