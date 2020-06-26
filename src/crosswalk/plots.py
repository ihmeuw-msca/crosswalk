def dose_response_curve(dose_variable, obs_method, continuous_variables=[],
                        mr_dir=None, mrdata=None, mrbrt=None,
                        file_name='dose_response_plot',
                        from_zero=False, include_bias=False,
                        ylim=None, y_transform=None, x_transform=None,
                        plot_note=None,
                        write_file=False):
    data_df = pd.DataFrame({'y': mrdata.obs, 'se': mrdata.obs_se, 'w': mrbrt.lt.w, 
                            f"{dose_variable}": dat1.covs[dose_variable], 
                            "obs_method": np.ravel(mrdata.alt_dorms),
                            "prev": mrdata.df['prev_alt'].values, 
                            "prev_se": mrdata.df['prev_se_alt'].values})
    
    # drop dose variable
    continuous_variables = [v for v in continuous_variables if v != dose_variable]
    
    # check for knots
    if dose_variable in mrdata.covs.columns:
        knots = mrbrt.cov_models[1].spline.knots
    else:
        knots = np.array([])
    
    if knots.any():
        min_cov = knots[0]
        max_cov = knots[-1]
    else:
        min_cov = np.min(data_df[dose_variable])
        max_cov = np.max(data_df[dose_variable])
        
    if from_zero:
        min_cov = 0

    # construct dataframe for prediction
    cov_range = (max_cov - min_cov)
    dose_grid = np.arange(min_cov, max_cov + cov_range * 0.01, cov_range / 100)

    cols = mrdata.covs.columns
    if include_bias:
        pred_df = pd.DataFrame(dict(zip(cols, np.ones(len(cols)))),
            index=np.arange(len(dose_grid)))
    else:
        pred_df = pd.DataFrame(dict(zip(cols, np.zeros(len(cols)))),
            index=np.arange(len(dose_grid)))
        pred_df['intercept'] = 1
        
    # if it's continuous variables, take median 
    for var in continuous_variables:
        pred_df[var] = np.median(mrdata.get_covs(var))

    # predict for line
    pred_df[dose_variable] = dose_grid
    pred_df['row_id'] = np.arange(1, len(pred_df)+1)
    pred_df['obs_method'] = obs_method
    pred_df['prev'] = 0.1
    pred_df['prev_se'] = 0.1

    # predict for mrdata
    y_pred = mrbrt.adjust_orig_vals(       
          df=pred_df,            
          orig_dorms = "obs_method", 
          orig_vals_mean = "prev",  
          orig_vals_se = "prev_se",
        data_id = 'row_id'
        )

    y_mean = y_pred['pred_diff_mean']
    y_sd_fixed = y_pred['pred_diff_sd']
#     y_pred = np.ravel(y_pred)
#     y_draws_random = np.random.normal(y_mean, y_sd, [1000, len(y_mean)]).T
#     y_draws_non_random = np.random.normal(y_mean, y_sd_fixed, [1000, len(y_mean)]).T
    gamma = mrbrt.gamma

    y_sd = np.sqrt(y_sd_fixed**2 + gamma)


    y_lo, y_hi = y_mean - 1.96*y_sd, y_mean + 1.96*y_sd
    y_lo_fe, y_hi_fe = y_mean - 1.96*y_sd_fixed, y_mean + 1.96*y_sd_fixed
#     y_lo, y_hi = np.quantile(y_draws_random, [0.025, 0.975], axis=1)
#     y_lo_fe, y_hi_fe = np.quantile(y_draws_non_random, [0.025, 0.975], axis=1)

     # predict for mrdata
    data_df['intercept'] = 1
    data_pred = mrbrt.adjust_orig_vals(       
          df=data_df,            
          orig_dorms = "obs_method", 
          orig_vals_mean = "prev",  
          orig_vals_se = "prev_se"
        )
    data_pred = data_pred['pred_diff_mean']
    # determine points inside/outside funnel
    data_df['position'] = 'inside funnel'
    data_df.loc[data_df.y < (data_pred - (data_df.se * 1.96)).values,
                'position'] = 'outside funnel'
    data_df.loc[data_df.y > (data_pred + (data_df.se * 1.96)).values,
                'position'] = 'outside funnel'
    # get inlier/outlier 
    data_df.loc[data_df.w >= 0.6, 'trim'] = 'inlier'
    data_df.loc[data_df.w < 0.6, 'trim'] = 'outlier'

    # get plot guide
    data_df['plot_guide'] = data_df['trim'] + ', ' + data_df['position']
    plot_key = {
        'inlier, inside funnel':('o', 'seagreen', 'darkgreen'),
        'inlier, outside funnel':('o', 'coral', 'firebrick'),
        'outlier, inside funnel':('x', 'darkgreen', 'darkgreen'),
        'outlier, outside funnel':('x', 'firebrick', 'firebrick')
    }
    
    # get scaled marker size
    data_df['size_var'] = 1 / data_df.se
    data_df['size_var'] = data_df['size_var'] * (300 / data_df['size_var'].max())
    
    # plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    plt.rcParams['axes.edgecolor'] = '0.15'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.fill_between(pred_df[dose_variable], y_lo, y_hi, alpha=0.5, color='lightgrey')
    plt.fill_between(pred_df[dose_variable], y_lo_fe, y_hi_fe, alpha=0.75, color='darkgrey')
    plt.plot(pred_df[dose_variable], y_mean, color='black', linewidth=0.75)
    plt.xlim([min_cov, max_cov])
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel('Exposure', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Effect size', fontsize=10)
    plt.yticks(fontsize=10)
    
    for key, value in plot_key.items():
        plt.scatter(
            data_df.loc[data_df.plot_guide == key, f'{dose_variable}'],
            data_df.loc[data_df.plot_guide == key, 'y'],
            s=data_df.loc[data_df.plot_guide == key, 'size_var'],
            marker=value[0],
            facecolors=value[1], edgecolors=value[2], linewidth=0.6,
            alpha=.6,
            label=key
        )
    betas = list(np.round(mrbrt.fixed_vars[obs_method], 3))
    content_string = f"betas: {betas}"
    if plot_note is not None:
        plt.title(content_string, fontsize=6)
        plt.suptitle(plot_note, y=1.01, fontsize=8)
    else:
        plt.title(content_string, fontsize=8)
    plt.legend(loc='upper left')
    
    for knot in knots:
        plt.axvline(knot, color='navy', linestyle='--', alpha=0.5, linewidth=0.75)
    
    if write_file:
        assert mrdir is not None, "mrdir is not specified!"
        assert file_name is not None, "file_name is not specified!"
        plt.savefig(os.path.join(mr_dir, f'{file_name}.pdf'), orientation='landscape', bbox_inches='tight')
    else:
        plt.show()


def funnel_plot(obs_method='Self-reported', mrdir=None, mrdata=None, mrbrt=None, 
                continuous_variables=[], file_name='funnel_plot', 
                plot_note=None, include_bias=False, write_file=False, 
                beta_samples=None, gamma_samples=None):
    """Funnel Plot."""
    data_df = pd.DataFrame({'y': mrdata.obs, 'se': mrdata.obs_se, 'w': mrbrt.lt.w})
    
    # inlier, outlier
    il_data_df = data_df.loc[data_df.w >= 0.6]
    ol_data_df = data_df.loc[data_df.w < 0.6]
        
    # construct dataframe for prediction
    pred_df = pd.DataFrame({'obs_method': obs_method,  
                            'prev': 0.9, 
                            'prev_se': 0.9}, index=[0])
    # if it's continuous variables, take median 
    for var in continuous_variables:
        pred_df[var] = np.median(mrdata.covs[var])
    
    # predict effect
    y_pred = mrbrt.adjust_orig_vals(       
          df=pred_df,            
          orig_dorms = "obs_method", 
          orig_vals_mean = "prev",  
          orig_vals_se = "prev_se"
        ).to_numpy()
    y_pred = np.ravel(y_pred)
    y_mean, y_sd = y_pred[2], y_pred[3]
    # create draws
    y_lower, y_upper = y_mean - 1.96*y_sd, y_mean + 1.96*y_sd
#     y_draws = np.random.normal(y_mean, y_sd, 1000)
    
    # triangle
    max_se = mrdata.obs_se.max()
    se_domain = np.arange(0, max_se*1.1, max_se / 100)
    se_lower = y_mean - (se_domain*1.96)
    se_upper = y_mean + (se_domain*1.96)
    
#     y_mean, y_lower, y_upper = np.round(y_mean, 3), np.round(y_lower, 3), np.round(y_upper, 3)
    
    # p-value
#     if y_mean > 0:
#         p_value = np.mean(y_draws < 0)/2.0
#     else:
#         p_value = np.mean(y_draws > 0)/2.0
    
    p_value = cw.utils.p_value(np.array([y_mean]), np.array([y_sd]))[0]
    
    content_string = f"Mean effect: {np.round(y_mean, 3)}\
    (95% CI: {np.round(y_lower, 3)} to {np.round(y_upper, 3)});\
    p-value: {np.round(p_value, 4)}"
    
    sns.set_style('darkgrid')
    plt.rcParams['axes.edgecolor'] = '0.15'
    plt.rcParams['axes.linewidth'] = 0.5
    plt.figure(figsize=(10,8))
    plt.fill_betweenx(se_domain, se_lower, se_upper, color='white', alpha=0.75)
    plt.axvline(y_mean, 0, 1 - (0.025*max(se_domain) / (max(se_domain)*1.025)), 
                color='black', alpha=0.75, linewidth=0.75)
    plt.plot(se_lower, se_domain, color='black', linestyle='--', linewidth=0.75)
    plt.plot(se_upper, se_domain, color='black', linestyle='--', linewidth=0.75)
    plt.ylim([-0.025*max(se_domain), max(se_domain)])
    plt.xlabel('Effect size', fontsize=10)
    plt.xticks(fontsize=10)
    plt.ylabel('Standard error', fontsize=10)
    plt.yticks(fontsize=10)
    plt.axvline(0, color='mediumseagreen', alpha=0.75, linewidth=0.75)
    plt.plot(il_data_df.y, il_data_df.se, 'o',
             markersize=5, markerfacecolor='royalblue', markeredgecolor='navy', markeredgewidth=0.6,
             alpha=.6, label='Inlier')
    plt.plot(ol_data_df.y, ol_data_df.se, 'o',
             markersize=5, markerfacecolor='indianred', markeredgecolor='maroon',  markeredgewidth=0.6,
             alpha=.6, label='Outlier')
    plt.legend(loc='upper left')
    plt.gca().invert_yaxis()
    if plot_note is not None:
        plt.title(content_string, fontsize=6)
        plt.suptitle(plot_note, y=1.01, fontsize=8)
    else:
        plt.title(content_string, fontsize=8)
        
    if write_file:
        assert mrdir is not None, "mrdir is not specified!"
        assert file_name is not None, "file_name is not specified!"
        plt.savefig(os.path.join(mrdir, file_name + '.pdf'), orientation='landscape', bbox_inches='tight')
    else:
        plt.show()
    plt.clf()
