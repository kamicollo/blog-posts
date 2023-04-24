import numpy as np
import pandas as pd


WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def simulate_users(n_users, parameters, rng = np.random.default_rng()):

    assert np.isclose(parameters['signup_weights'].sum(),1), f"Signup weights do not add up to 1 (sum is {(parameters['signup_weights'].sum()):2f})"
    assert np.isclose(parameters['newsletter_day_weights'].sum(),1), f"Newsletter day weights do not add up to 1 (sum is {(parameters['summary_day_weights'].sum()):2f})"

    user_signup_day = rng.choice(parameters.index, p = parameters['signup_weights'], size=n_users)
    user_summary_day = rng.choice(parameters.index, p = parameters['newsletter_day_weights'], size=n_users)

    parameters['intrinsic_motivation'] = [(b[0] + a[0], b[1] + a[1]) for (b,a) in zip(
        parameters['intrinsic_motivation_baseline'].values, parameters['intrinsic_motivation_adj'].values
    )]

    a, b = zip(*parameters['intrinsic_motivation'][user_signup_day].values)

    user_motivation = rng.beta(a=a, b=b, size = n_users)

    return pd.DataFrame({
        'signup_day': user_signup_day,
        'newsletter_day': user_summary_day,        
        'intrinsic_motivation': user_motivation
    })


def simulate_activity(users, behavior_parameters, rng = np.random.default_rng()):

    tenure_days = behavior_parameters['tenure_length']
    days = np.arange(tenure_days)
    if len(behavior_parameters['full_tenure_impact']) == 1:
        behavior_parameters['full_tenure_impact'] = behavior_parameters['full_tenure_impact'] * 7

    #tenure_impact = behavior_parameters['tenure_curve'](days) * behavior_parameters['full_tenure_impact']
    tenure_impact = behavior_parameters['tenure_curve'](days).reshape(-1, 1) * np.array(behavior_parameters['full_tenure_impact']).reshape(1, -1)

    day_dict = {name: num for num,name in enumerate(WEEKDAYS)}

    user_signup_day_nums = np.array([day_dict[d] for d in users['signup_day']])    
    
    user_days_of_week = (
        user_signup_day_nums.reshape(-1, 1).repeat(tenure_days, axis=1) + 
        np.arange(tenure_days).reshape(1, -1)
    ) % 7

    #initial point - intrinsic motivation
    user_matrix = np.zeros((len(users), tenure_days))
    user_matrix += users['intrinsic_motivation'].values.reshape(-1, 1)

    #add weekday impact
    user_matrix += behavior_parameters['weekday_impact'][user_days_of_week]

    #add summary day impact

    user_summary_day_nums = np.array([day_dict[d] for d in users['newsletter_day']])
    user_matrix += (
        behavior_parameters['newsletter_day_impact'] * 
        (user_summary_day_nums.reshape(-1, 1).repeat(tenure_days, axis=1) == user_days_of_week)
    )

    #add tenure effect     
    #user_matrix += tenure_impact.reshape(1, -1).repeat(len(users), axis=0)
    user_matrix += tenure_impact[:,user_signup_day_nums].T

    #how many out-of-range probabilities do we get?
    pct_cutoff = np.sum((user_matrix <= 0.01) | (user_matrix >= 0.99)) / user_matrix.size

    #ensure probabilities are within 0-1 range
    user_matrix[user_matrix <= 0.01] = 0.01
    user_matrix[user_matrix >= 0.99] = 0.99
    
    #simulate activity & return
    activity =  rng.binomial(1, user_matrix)

    #convert to pandas df

    #first, convert to a list of lists, a list per user
    observations = zip(
        user_days_of_week.tolist(),
        activity.tolist()
    )

    #get inverse day name mapping
    inverse_day_dict = {num: name for name,num in day_dict.items()}

    #convert to observation-level data
    user_day_visits = [
        list(
            zip(np.arange(tenure_days) + 1, [inverse_day_dict[d] for d in days], visits)
        ) for (days, visits) in observations
    ]

    activity_df = pd.DataFrame(
        zip(user_day_visits, users['signup_day'], users['newsletter_day']),
        columns = ['visits', 'signup_day_of_week', 'newsletter_day_of_week']
    ).explode('visits').reset_index(names=['user_id'])

    activity_df[['tenure_day', 'day_of_week', 'visit']] = activity_df['visits'].to_list()

    activity_df['is_newsletter_day'] = (activity_df['day_of_week'] == activity_df['newsletter_day_of_week']) * 1

    cat_type = pd.api.types.CategoricalDtype(categories=WEEKDAYS, ordered=True)
    activity_df['day_of_week'] = activity_df['day_of_week'].astype(cat_type)
    activity_df['newsletter_day_of_week'] = activity_df['newsletter_day_of_week'].astype(cat_type)
    activity_df['signup_day_of_week'] = activity_df['signup_day_of_week'].astype(cat_type)

    return activity_df.drop(['visits'], axis=1), pct_cutoff
