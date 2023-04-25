import pandas as pd
import altair as alt
import scipy.special as scs
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def basic_eda(activity):
    by_tenure_day = pd.DataFrame(activity.groupby('tenure_day')['visit'].mean()).reset_index()
    by_weekday = pd.DataFrame(activity.groupby('day_of_week')['visit'].mean()).reset_index()
    by_signup_day = pd.DataFrame(activity.groupby('signup_day_of_week')['visit'].mean()).reset_index()

    c1 = alt.Chart(by_tenure_day).mark_line().encode(
        x=alt.X("tenure_day:O", title="Tenure day", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("visit", title='% active users')
    ).properties(title='Drop-off curve', height=150)

    c2 = alt.Chart(by_weekday).mark_line().encode(
        x=alt.X("day_of_week:O", sort=WEEKDAYS, title='Day of week', axis=alt.Axis(labelAngle=0)),
        y=alt.Y("visit", title='% active users')
    ).properties(title='% active users by day of the week', height=150)

    c3 = alt.Chart(by_signup_day).mark_line().encode(
        x=alt.X("signup_day_of_week:O", sort=WEEKDAYS, title='Signup day of the week', axis=alt.Axis(labelAngle=0)),
        y=alt.Y(
            "visit", 
            scale=alt.Scale(
                domain=[by_signup_day['visit'].min()*0.9, by_signup_day['visit'].max()*1.1]
            ),
            title='% active days'
        )
    ).properties(title='% Active days by user signup day', height=150)

    return c1, c2, c3

def plot_effects(df, var_name, title, sort="ascending", y_title="Δ probability", domain=None):
    df = df[df['variable'] == var_name]
    has_series = (df['values'].isna().sum() == 0)
        
    dow_chart = alt.Chart(df)    
    if domain is None:
        d_range = max(df['higher'].max() - df['lower'].min(), 0.1)        
        add_scale = 0.1
        domain = [ -add_scale * d_range + df['lower'].min(), df['higher'].max() + add_scale * d_range ]

    points = dow_chart.mark_point().encode(
        alt.X('values' if has_series else 'source', sort=sort, title=""),
        alt.Y('impact', title=y_title, scale=alt.Scale(domain=domain)),
        alt.Color('source'),
    )

    if has_series:
        final_chart = points + dow_chart.mark_line().encode(
            alt.X('values', sort=sort, title=""),
            alt.Y('impact', title=y_title, scale=alt.Scale(domain=domain)),
            alt.Color('source'),
        ) + dow_chart.mark_area(opacity=0.5).encode(
            alt.X('values', sort=sort, title=""),
            alt.Y('lower', title=""),
            alt.Y2('higher', title=''),
            alt.Color('source'),
        )
    else:
        final_chart = points + dow_chart.mark_errorbar().encode(
            alt.X('source', title=""),
            alt.Y('lower', title="", scale=alt.Scale(domain=domain)),
            alt.Y2('higher', title=''),
            alt.Color('source'),
        )

    return final_chart.properties(title=title, height=100)


def plot_all_params(all_params):

    return alt.hconcat(
        plot_effects(all_params, 'signup_weights', 'Day of week signup weights', y_title='weight', domain=[0,1], sort=WEEKDAYS),
        plot_effects(all_params, 'newsletter_day_weights', 'Newsletter day weights', y_title='weight', domain=[0,1], sort=WEEKDAYS),
        plot_effects(all_params, 'motivation', 'Baseline motivation', y_title="Probability to visit"),
    ) & alt.hconcat(    
        plot_effects(all_params, 'day_of_week', 'Day of week effects', sort=WEEKDAYS),
        plot_effects(all_params, 'signup_day_of_week', 'Signup day of week effects', sort=WEEKDAYS),        
        plot_effects(all_params, 'is_newsletter_day', 'Newsletter day effects'),
    ) & alt.hconcat(
        plot_effects(all_params, 'tenure_day', 'Tenure effects').properties(width=400),
    )

def visualize_tenure_prior(prior, name, subtitle, prior_set):
    α = pm.Normal.dist(*prior_set['α'])
    β = pm.Normal.dist(*prior_set['β'])
    
    p = pm.Dirichlet.dist(a = prior)
    A, B, P = pm.draw([α, β, p], 100)

    T_eff = scs.expit(A.reshape(-1,1) + B.reshape(-1, 1) * np.cumsum(P, axis=1))
    T_eff_mean = T_eff.mean(axis=0)

    lines = pd.DataFrame(T_eff).melt(ignore_index=False).reset_index(names=['draw'])
    mean_line = pd.DataFrame(T_eff_mean).melt(ignore_index=False).reset_index(names=['xx']).rename({'variable': 'draw', 'xx': 'variable', }, axis=1)

    return alt.Chart(lines).mark_line(opacity=0.5).encode(
        alt.X('variable'), alt.Y('value'), alt.Detail('draw')
    ) + alt.Chart(mean_line).mark_line(color='red').encode(
        alt.X('variable', title='Tenure day'), alt.Y('value', title='Visit probability'), alt.Detail('draw')
    ).properties(title={'text': name, 'subtitle': subtitle}, width=150, height=150) 

def plot_tenure_priors(params, prior_set):
    return alt.vconcat(
        alt.hconcat(
            visualize_tenure_prior(
                np.ones(params['tenure_length'] - 1) * 10, 
                name='Flat prior (10,10,10...10,10)', subtitle='Relatively smooth curves',
                prior_set=prior_set
            ),        
            visualize_tenure_prior(
                np.ones(params['tenure_length'] - 1), 
                name='Flat prior (1,1,1...1,1)', subtitle='Less smooth curves',
                prior_set=prior_set
            ),
            visualize_tenure_prior(
                np.ones(params['tenure_length'] - 1) / 10, 
                name='Flat prior (0.1,0.1,0.1...0.1,0.1)', subtitle='Jagged curves',
                prior_set=prior_set
            ),
            
        ),
        alt.hconcat(
            visualize_tenure_prior(
                (1 + np.arange(params['tenure_length'] - 1)), 
                name='Increasing prior (1,2,3..28)', subtitle='Smooth curves, late drop-off',
                prior_set=prior_set
            ),
            visualize_tenure_prior(
                (1 + np.arange(params['tenure_length'] - 1)) / 28, 
                name='Increasing prior (1/28,2/28,3/28..1)', subtitle='Jagged curves, late drop-off',
                prior_set=prior_set
            ),        
            visualize_tenure_prior(
                (1 + np.arange(params['tenure_length'] - 1))[::-1], 
                name='Decreasing prior (28, 27,26..1)', subtitle='Smooth curves, early drop-off',
                prior_set=prior_set
            ),        
            
        )
)

def plot_rhats(trace):
    var_names = ['α', 'β', 'newsletter_day', 'weekday_effect', 'signup_day_effect', 'σ_dow', 'σ_signup_dow', 'dow_correlation']   
    max_rhat = pd.DataFrame(az.rhat(
        trace, 
        var_names=[v for v in var_names if v in list(trace['posterior'].keys())],
    ).to_dataframe().apply(max, axis=0), columns=['rhat']).reset_index(names='variable')

    c = alt.Chart(max_rhat).encode(
        alt.Y('variable'), alt.X('rhat'), alt.Text('rhat', format='.2f')
    )

    return c.mark_bar() + c.mark_text(align='left', dx=2)

def plot_trace(trace):
    var_names = ['α', 'β', 'newsletter_day', 'weekday_effect', 'signup_day_effect', 'σ_dow', 'σ_signup_dow', 'dow_correlation', 'avg_motivation']    
    az.plot_trace(
        trace, 
        var_names=[v for v in var_names if v in list(trace['posterior'].keys())],
        figsize=(10,10)
    )
    plt.tight_layout()
    plt.show()


def plot_all_effects(all_params):

    return alt.hconcat(    
        plot_effects(all_params, 'motivation', 'Baseline motivation', y_title="Probability to visit"),
        plot_effects(all_params, 'day_of_week', 'Day of week effects', sort=WEEKDAYS),
    ) & alt.hconcat(
        plot_effects(all_params, 'signup_day_of_week', 'Signup day of week effects', sort=WEEKDAYS),        
        plot_effects(all_params, 'is_newsletter_day', 'Newsletter day effects'),
    ) & alt.hconcat(
        plot_effects(all_params, 'tenure_day', 'Tenure effects').properties(width=300, height=150),
    )


