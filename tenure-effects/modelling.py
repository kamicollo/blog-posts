import pandas as pd
import numpy as np
import statsmodels.api as sm
import pymc as pm
from pymc.sampling_jax import sample_blackjax_nuts



def make_groups(activity):
    grouped = pd.DataFrame(
        activity.groupby(
            ['tenure_day', 'day_of_week', 'is_newsletter_day', 'signup_day_of_week']
        ).apply(lambda x: (x['visit'].count(), x['visit'].sum()))
    ).reset_index()

    grouped[['total_users', 'visited']] = grouped[0].to_list()    

    grouped = grouped[['day_of_week', 'tenure_day', 'visited', 'total_users', 'is_newsletter_day', 'signup_day_of_week']]
    grouped['not_visited'] = grouped['total_users'] - grouped['visited']

    return grouped

def prep_for_stats_models(df, discrete_tenure=True, ind_effects=False):
    Y = df[['visited', 'not_visited']]
    if ind_effects:
        exog = df[['day_of_week', 'tenure_day', 'is_newsletter_day', 'signup_day_of_week', 'user_id']].copy()
        exog['user_id'] = exog['user_id'].astype('string')
    else:
        exog = df[['day_of_week', 'tenure_day', 'is_newsletter_day', 'signup_day_of_week']].copy()
    if discrete_tenure:
        exog['tenure_day'] = exog['tenure_day'].astype('string')
    else:
        exog['tenure_day'] = exog['tenure_day'].astype('int')

    X = sm.add_constant(pd.get_dummies(exog, drop_first=True))
    return X,Y

def fit_mle(df, discrete_tenure = True, ind_effects=False):    
    X, Y = prep_for_stats_models(df, discrete_tenure=discrete_tenure, ind_effects=ind_effects)
    glm_binom = sm.GLM(Y, X, family=sm.families.Binomial())
    res = glm_binom.fit()
    return res

def fit_pymc(model, rng, chains=4, **kwargs):
    import sys
    stdout = sys.stdout
    sys.stdout = None
    try:
        return sample_blackjax_nuts(random_seed=rng, model = model, idata_kwargs={"log_likelihood": True}, chains=chains, **kwargs) 
    finally:
        sys.stdout = stdout        

def compile_pymc(group_df, hierarchical = True, priors = {}, model_with_offset = False, multivariate_b = False):
    dow_idx, days_of_week = pd.factorize(group_df['day_of_week'])
    t_idx, tenure_days = pd.factorize(group_df['tenure_day'])
    signup_dow_idx, sdow = pd.factorize(group_df['signup_day_of_week'])

    assert np.all(days_of_week == sdow)
    if multivariate_b:
        features = ['day_of_week', 'signup_day', 'tenure']
        mu_priors = [priors['S'][0], priors['S'][0], priors['β'][0]]
        sigma_priors = [priors['S'][1], priors['S'][1], priors['β'][1]]
    else:
        features = ['day_of_week', 'signup_day']
        mu_priors = [priors['S'][0], priors['S'][0]]
        sigma_priors = [priors['S'][1], priors['S'][1]]

    coords= {
        'day_of_week': days_of_week,    
        'day_of_week_less_one': days_of_week[1:],
        'tenure_day': tenure_days,
        'tenure_days_less_1': tenure_days[1:],
        'features': features,
        "groups": group_df.index
    }

    with pm.Model(coords=coords) as model:
    
        α = pm.Normal("α", priors['α'][0], priors['α'][1]) #main intercept        
        
        
        #multivariate spec
        if hierarchical:            
            σ_dow = pm.HalfNormal.dist(sigma_priors, shape=len(features)) #variance of weekday & signup day effects
            chol, corr, stds = pm.LKJCholeskyCov('chol_cov', n=len(features), eta=priors['η'], sd_dist=σ_dow, compute_corr=True)

            if model_with_offset:
                Z = pm.Normal('Z', mu=mu_priors, sigma=1, dims=["day_of_week_less_one", "features"])
                vals_1 = pm.math.dot(Z, chol)
                vals = pm.Deterministic("fts", pm.math.concatenate([np.zeros((1, len(features))), vals_1], axis=0), dims=['day_of_week', 'features'])

            else:
                Z = pm.Normal('Z', mu=mu_priors, sigma=1, dims=["day_of_week", "features"])
                vals = pm.math.dot(Z, chol)

        else:
            corr = None
            stds = None

            if model_with_offset:
                vals_1 = pm.Normal('fts_1', mu_priors, sigma_priors, dims=['day_of_week_less_one', 'features'])
                vals = pm.Deterministic("fts", pm.math.concatenate([np.zeros((1, len(features))), vals_1], axis=0), dims=['day_of_week', 'features'])
            else: 
                vals = pm.Normal('fts_1', mu_priors, sigma_priors, dims=['day_of_week', 'features'])
            
            
        W = pm.Deterministic("weekday_effect", vals[:, 0], dims="day_of_week")
        S = pm.Deterministic("signup_day_effect", vals[:, 1], dims="day_of_week") 
        N = pm.Normal("newsletter_day", priors['N'][0], sigma=priors['N'][1])

        T = pm.Dirichlet("tenure_prop", a=priors['p'], dims='tenure_days_less_1')

        if multivariate_b:
            β = pm.Deterministic("β", vals[:, 2], dims="day_of_week").reshape((-1, 1))
            day_props = pm.math.concatenate([np.zeros(1), T.cumsum(axis=0)], axis=0).T.reshape((1, -1))
        
            cumT = pm.Deterministic(
                "tenure_day_impact", 
                pm.math.dot(β, day_props), 
                dims=['day_of_week', 'tenure_day']
            )            

            tenure_impact = cumT[signup_dow_idx, t_idx]            

        else:
            β = pm.Normal("β", mu=priors['β'][0], sigma=priors['β'][1])
            cumT = pm.Deterministic(
                "tenure_day_impact", 
                β * pm.math.concatenate([np.zeros(1), T.cumsum(axis=0)], axis=0), 
                dims='tenure_day'
            )

            tenure_impact = cumT[t_idx]
        
        inv_logit = α + W[dow_idx] + N*group_df['is_newsletter_day'] + tenure_impact + S[signup_dow_idx]
        pm.Binomial("likelihood", n = group_df['total_users'], logit_p = inv_logit, observed = group_df['visited'])

        add_marginal_effects(α, cumT, W, S, N, stds, corr, multivariate_b)

    return model




def add_marginal_effects(α, cumT, W, S, N, stds, corr, multivariate_b):
    #deterministic marginal effects

    if corr:
        pm.Deterministic("dow_correlation", corr[0,1])
        if multivariate_b:            
            pm.Deterministic("signup_tenure_correlation", corr[1,2])
    if stds:    
        pm.Deterministic("σ_dow", stds[0])
        pm.Deterministic("σ_signup_dow", stds[1])
        if multivariate_b:
            pm.Deterministic("σ_tenure_dow", stds[2])

    baseline = pm.Deterministic("avg_motivation", pm.math.invlogit(α))

    if multivariate_b:
        tenure_dims = ['day_of_week', 'tenure_day']
    else:
        tenure_dims = 'tenure_day'
        
    pm.Deterministic(
        "δtenure_impact", 
        (pm.math.invlogit(α + cumT) - baseline),
        dims=tenure_dims
    )

    pm.Deterministic(
        "δweekday_impact",
        (pm.math.invlogit(α + W) - baseline),
        dims='day_of_week'
    )

    pm.Deterministic(
        "δsignup_day_impact",
        (pm.math.invlogit(α + S) - baseline),
        dims='day_of_week'
    )

    pm.Deterministic(
        "δnewsletter_day_impact",
        (pm.math.invlogit(α + N) - baseline)        
    )

    
    ## with offsets
    if not multivariate_b:
        
        baseline_offset = α + cumT[0] + W[0] + S[0]
        baseline_offset_p = pm.math.invlogit(baseline_offset)

        pm.Deterministic(
            "δtenure_impact_offset", 
            pm.math.invlogit(cumT + α + W[0] + S[0]) - baseline_offset_p,
            dims='tenure_day'
        )

        pm.Deterministic(
            "δweekday_impact_offset",
            pm.math.invlogit(W + α + cumT[0] + S[0]) - baseline_offset_p,
            dims='day_of_week'
        )

        pm.Deterministic(
            "δsignup_day_impact_offset",
            pm.math.invlogit(S + α + W[0] + cumT[0]) - baseline_offset_p,
            dims='day_of_week'
        )  

        pm.Deterministic(
            "δnewsletter_day_impact_offset",
            pm.math.invlogit(N + baseline_offset) - baseline_offset_p,
        ) 