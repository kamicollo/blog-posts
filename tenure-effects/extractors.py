import scipy.special as scs
import numpy as np
import pandas as pd
import arviz as az

def parse_parameters(user_params, activity_params, name, weekdays):
    tpe = TrueParameterExtractor(user_params, activity_params, name, weekdays)
        
    params = pd.concat([
        tpe.extract_dow_effects(offset=False), tpe.extract_newsletter_effects(), tpe.extract_signup_effects(offset=False), 
        tpe.extract_tenure_effects(offset=False), tpe.extract_motivation_effects(),
        tpe.extract_full_tenure_effects(offset=False)
    ], axis=0)

    weights = user_params[['signup_weights', 'newsletter_day_weights']].melt(ignore_index=False).reset_index()
    weights['source'] = name
    weights = weights.rename({'value': 'impact', 'weekday': 'values'}, axis=1)
    weights['higher'] = weights['impact']
    weights['lower'] = weights['impact']
    
    return pd.concat([weights, params], axis=0)

class StatsModelExtractor():
    def __init__(self, res, name='Statsmodels', tenure_discrete=True) -> None:
        self.res = res
        self.name = name
        self.tenure_discrete = tenure_discrete
        self.results = res.summary2().tables[1]
        self.intercept = res.params['const'] if tenure_discrete else res.params['const'] + res.params['tenure_day']
        self.marginal_effects = scs.expit(self.results + self.intercept) - scs.expit(self.intercept)    
    
    def extract_effects(self, var_name):
        parsed = [(d[(len(var_name) + 1):], d) for d in self.marginal_effects.index if d.startswith(var_name)]
        dim_values, idx = zip(*parsed)        
        effects = self.marginal_effects.loc[idx, ['Coef.', '[0.025', '0.975]']]
        effects.columns = ['impact', 'lower', 'higher']        
        effects['source'] = self.name
        effects['values'] = [None if d == "" else d for d in dim_values]
        effects['variable'] = var_name
        return effects.reset_index(drop=True)

    
    def extract_newsletter_effects(self):
        return self.extract_effects('is_newsletter_day')   
    
    def extract_tenure_effects(self, day_range = np.arange(27) + 2):
        if self.tenure_discrete:
            df = self.extract_effects('tenure_day')
            df['values'] = df['values'].astype(int)
            return df
        else:
            df = pd.DataFrame({
                'source': self.name, 
                'variable': 'tenure_day', 
                'values': day_range,
                'impact': 0,
                'higher': 0,
                'lower': 0
            })

            df['impact'] = scs.expit(self.res.params['tenure_day'] * df['values'] + self.res.params['const']) - scs.expit(self.intercept)
            df['higher'] = scs.expit(
                self.results.loc['const', '0.975]'] + self.results.loc['tenure_day', '0.975]'] * df['values']
                ) - scs.expit(
                    self.results.loc['const', '0.975]'] + self.results.loc['tenure_day', '0.975]']
                )
            df['lower'] = scs.expit(
                self.results.loc['const', '[0.025'] + self.results.loc['tenure_day', '[0.025'] * df['values']
                ) - scs.expit(
                    self.results.loc['const', '[0.025'] + self.results.loc['tenure_day', '[0.025']
                )
            return df
    
    def extract_signup_effects(self):
        return self.extract_effects('signup_day_of_week')     

    def extract_dow_effects(self):
        return self.extract_effects('day_of_week')     
    
    def extract_all_effects(self):
        return pd.concat([
            self.extract_dow_effects(),            
            self.extract_newsletter_effects(),
            self.extract_signup_effects(),
            self.extract_tenure_effects()
        ], axis=0)
    

class TrueParameterExtractor():
    def __init__(self, user_params, activity_params, name = 'Ground truth', weekdays = range(7)) -> None:
        self.user_params = user_params
        self.activity_params = activity_params
        self.name = name
        self.weekdays = weekdays

    def to_df(self, values, var_name, dim=None):
        return pd.DataFrame({
            'impact': values,
            'higher': values,
            'lower': values,
            'variable': var_name,
            'values': dim,
            'source': self.name
        })

    
    def extract_newsletter_effects(self):
        return self.to_df([self.activity_params['newsletter_day_impact']], 'is_newsletter_day')
    
    def extract_tenure_effects(self, offset=True, multi=False):
        days = np.arange(self.activity_params['tenure_length'])
        if not multi:
            tenure_curve = self.activity_params['tenure_curve'](days) * (self.activity_params['full_tenure_impact'][0])
            if offset:
                impact = tenure_curve[1:] - tenure_curve[0]
            else:
                impact = tenure_curve

            return self.to_df(impact, 'tenure_day', (days + 1)[(offset):])
        else: 
            tenure_curve = (
                self.activity_params['tenure_curve'](days).reshape(-1, 1) * np.array(self.activity_params['full_tenure_impact']).reshape(1, -1)
            ).T
            if offset:
                impact = tenure_curve[1:,:] - tenure_curve[0,:]
            else:
                impact = tenure_curve
            
            dfs = []
            for i, w in enumerate(self.weekdays[(1*offset):]):

                df = self.to_df(impact[i,(offset):], 'tenure_day', (days + 1)[(offset):])
                df['factor'] = w
                dfs.append(df)
            return pd.concat(dfs)
        
    
    def extract_full_tenure_effects(self, offset=True):
        if offset:
            impact = self.activity_params['full_tenure_impact'][1:] - self.activity_params['full_tenure_impact'][0]
        else:
            impact = self.activity_params['full_tenure_impact']

        return self.to_df(impact, 'full_tenure_impact', self.weekdays[(1*offset):])


    
    def extract_dow_effects(self, offset=True):
        if offset:
            impact = self.activity_params['weekday_impact'][1:] - self.activity_params['weekday_impact'][0]
        else: 
            impact = self.activity_params['weekday_impact']
        
        return self.to_df(impact, 'day_of_week', self.weekdays[(1*offset):])
    
    def extract_signup_effects(self, offset=True):
        baseline = [x[0]/sum(x) if sum(x) > 0 else 0 for x in self.user_params['intrinsic_motivation_baseline']]
        adj = [x[0]/sum(x) if sum(x) > 0 else 0 for x in self.user_params['intrinsic_motivation']]
        effect = [a - b for a,b in zip(adj, baseline)]
        impact = effect[1:] if offset else effect
        return self.to_df(impact, 'signup_day_of_week', self.weekdays[(1*offset):])
    
    def extract_motivation_effects(self):
        params = self.user_params['intrinsic_motivation_baseline'][0]
        return self.to_df([params[0] / sum(params)], 'motivation')
    
    def extract_all_effects(self, offset=True):
        return pd.concat([
            self.extract_dow_effects(offset=offset),
            self.extract_motivation_effects(),
            self.extract_newsletter_effects(),
            self.extract_signup_effects(offset=offset),
            self.extract_tenure_effects(offset=offset)
        ], axis=0)
    

class TraceExtractor():
    def __init__(self, trace, name = 'Bayesian') -> None:
        self.trace = trace
        self.name = name

    def extract_effect(self, var_name, dim, nice_name):
        if dim:
            hdi = az.hdi(self.trace, var_names=[var_name], hdi_prob=0.95).to_dataframe().reset_index().pivot(
                index=dim,columns="hdi", values=var_name
            )
            mean = self.trace['posterior'][var_name].mean(dim=['chain', 'draw']).to_dataframe()
        
            inferred_impact = hdi.join(mean).reset_index().rename(
                {
                var_name: 'impact',  
                dim: 'values'  
                }, axis=1
            )
        else:
            hdi = az.hdi(self.trace, var_names=[var_name], hdi_prob=0.95)
            lower, higher = hdi[var_name].to_numpy()

            inferred_impact = pd.DataFrame({
                'impact': self.trace['posterior'][var_name].mean(dim=['chain', 'draw']).to_numpy(),
                'higher': higher,
                'lower': lower,
                'values': None
            }, index=[0])            

        
        inferred_impact['source'] = self.name
        inferred_impact['variable'] = nice_name
        return inferred_impact


    def extract_dow_effects(self, offset=False):
        vals = list(self.trace['posterior'].coords['day_of_week'].values)
        name = 'δweekday_impact'

        if offset:
            name = name + '_offset'
            vals = vals[1:]
        
        df = self.extract_effect(name, 'day_of_week', 'day_of_week')
        return df.set_index('values').loc[vals].reset_index()
    
    def extract_tenure_effects(self, offset=False):
        vals = list(self.trace['posterior'].coords['tenure_day'].values)
        name = 'δtenure_impact'

        if offset:
            name = name + '_offset'
            vals = vals[1:]
        
        df = self.extract_effect(name, 'tenure_day', 'tenure_day')
        return df.set_index('values').loc[vals].reset_index()        

    def extract_signup_effects(self, offset=False):
        vals = list(self.trace['posterior'].coords['day_of_week'].values)
        name = 'δsignup_day_impact'

        if offset:
            name = name + '_offset'
            vals = vals[1:]
        
        df = self.extract_effect(name, 'day_of_week', 'signup_day_of_week')
        return df.set_index('values').loc[vals].reset_index()                
    
    def extract_newsletter_effects(self, offset=False):
        name = 'δnewsletter_day_impact_offset' if offset else 'δnewsletter_day_impact'
        return self.extract_effect('δnewsletter_day_impact', None, 'is_newsletter_day')
    
    def extract_motivation_effects(self):
        return self.extract_effect('avg_motivation', None, 'motivation')
    
    def extract_all_effects(self, offset=True):
        return pd.concat([
            self.extract_dow_effects(offset=offset),
            self.extract_motivation_effects(),
            self.extract_newsletter_effects(offset=offset),
            self.extract_signup_effects(offset=offset),
            self.extract_tenure_effects(offset=offset)
        ], axis=0)
        