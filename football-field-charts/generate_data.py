
#%%
import numpy as np
import pandas as pd
import seaborn as sb
import datetime
import matplotlib.pyplot as plt
from faker import Faker

np.random.seed(1000)

id_range = np.arange(1, 100000, 1)


#price change sample; mean = 30*0.7 - 15 = 6; minimum = -15
price_changes = np.random.gamma(30, 0.7, 10000) - 15

sqm_changes = np.multiply(
    np.random.binomial(1, 0.2, 1000),
    np.random.gamma(30, 0.5, 1000) - 15
)

#%%

#create customers
no_customers, avg_contracts_per_customer = 100, 3
#to be used to adjust pricing afterwards
customer_pricing_segment = ['low', 'low', 'low', 'medium', 'medium', 'high']

def create_customers(no_customers, avg_contracts_per_customer, customer_pricing_segment):
    return pd.DataFrame({
        'name': [Faker('de_DE').company() for i in range(no_customers)],
        'customer_id': ["C-" + str(i) for i in np.random.choice(id_range, no_customers, False)],
        'no_contracts': np.random.poisson(avg_contracts_per_customer, no_customers),
        'pricing_segment': np.random.choice(customer_pricing_segment, no_customers, True)
    })

customers = create_customers(no_customers, avg_contracts_per_customer, customer_pricing_segment)
#%%

#create locations
no_locations = int(np.round(no_customers * avg_contracts_per_customer * 0.8))
min_construction_year, max_construction_year = 2005, 2025
avg_months_construction_to_contract = 12
peak_period = 2030

def create_locations(no_locations, min_c, max_c, peak_period, avg_to_contract):
    c_year_draws = [yr for yr in np.random.poisson(peak_period, 1000) if yr >= min_c and yr <= max_c]
    return pd.DataFrame({
        'location_id' : ["L-" + str(i) for i in np.random.choice(id_range, no_locations, False)],
        'address': [Faker('de_DE').street_address() for i in range(no_locations)],
        'construction_date': map(
            datetime.date,
            np.random.choice(c_year_draws, no_locations, True),
            np.random.randint(1, 13, no_locations), #starts any month
            1 + 14 * np.random.binomial(1, 0.5, no_locations) # day 1 or 15 only
        ),
        'time_to_first_contract': np.random.poisson(avg_to_contract, no_locations),
        'location': np.repeat({'lat': 55., 'lon': 22.}, no_locations)
    })

locations = create_locations(no_locations, min_construction_year, 
max_construction_year, peak_period, avg_months_construction_to_contract)

#%%
#create contracts and terms

max_age = 50 #in years
initial_term_length, mean_extension_length = 5*4, 4 #in quarters
mean_extensions = 3
min_sqm, mean_sqm, = 200., 750.
mean_price = 25

#simulate cyclical real estate market patterns
def yearly_price_adjustment(year):
    return 0.05 * np.sin(year / np.random.randint(1,3)) + (year - 2000) * 0.005

def get_price(start_date, customer_id):
    #initial price as poisson to be close to normal but avoid non-positive numbers (scaled)
    price = np.random.poisson(mean_price * 10) / 10
    #year adjustment
    price = price * (1 + yearly_price_adjustment(start_date.year))
    #customer segment adjustment
    segment = customers.query('customer_id == @customer_id').pricing_segment.values[0]
    if segment == 'low':
        price = price * (1 - np.random.normal(0.1, 0.02))
    elif segment == 'high':
        price = price * (1 + np.random.normal(0.1, 0.02))
    return price


def create_contract(loc, start_date, initial_term_length, mean_extensions, min_sqm, mean_sqm, customer_id):
    initial_term = (1 + np.random.poisson(initial_term_length - 1)) * 4    
    no_extensions = np.random.poisson(mean_extensions)
    sqm = min_sqm + np.random.beta(2, 5) * (mean_sqm - min_sqm) * (2 + 5) / 2
    price = get_price(start_date, customer_id)
    return {
        'initial_term' : initial_term,
        'start_date': start_date,
        'sqm': np.around(sqm),
        'price': np.around(price, 2),
        'extensions' : no_extensions,
        'customer_id': customer_id,
        'location_id': loc["location_id"],
    }

create_contract(locations.iloc[0], datetime.date(2020, 1, 1), 50, 3, 200, 500, customers.iloc[0].customer_id)

#%%

states = (['keep', 'change', 'idle'], [0.7, 0.2, 0.1]) #what happens when contract comes to end
mean_idle_time = 12 #months

def draw_customer():
    probs = customers.no_contracts / sum(customers.no_contracts)
    return np.random.choice(customers.customer_id, None, True, probs)

#price changes - mean of 1.5%, up to -1.5% negative
price_changes = np.random.beta(3, 7, 10000) / 10 - 0.015
#sqm changes - 10% chance that SQM changes; when it does, it's an equal chance of +-10% / 20% / 50% 
sqm_changes = np.append(np.repeat([0], 54), [0.1, 0.2, 0.5, -0.1, -0.2, -0.5])

def get_terms(contract):
    start_date = contract["start_date"]
    max_end_date = contract["end_of_life"]
    end_date = start_date + pd.DateOffset(months=contract["initial_term"]) - pd.DateOffset(days=1)
    price = contract["price"]
    sqm = contract["sqm"]
    term_table = [{        
        'extension': 0,
        'start': start_date,
        'end': end_date,
        'monthly_rent': price * sqm,
        'sqm': sqm
    }]
    for i in range(contract["extensions"]):
        if end_date < max_end_date:
            start_date = end_date + pd.DateOffset(days=1)
            term_length = (1 + np.random.poisson(mean_extension_length - 1)) * 4
            end_date = start_date + pd.DateOffset(months=term_length) - pd.DateOffset(days=1)
            price = round(price * (1 + np.random.choice(price_changes)) * 100) / 100
            sqm = sqm * (1 + np.random.choice(sqm_changes))
            temp = {                
                'extension': i + 1,
                'start': start_date,
                'end': end_date,
                'monthly_rent': price * sqm,
                'sqm': sqm
            }
            term_table.append(temp)
    return term_table

def create_contracts(locs, max_forecast_year, initial_term_length, mean_extensions, min_sqm, mean_sqm):
    contracts = []    
    for loc in locs.to_dict(orient="records"):
        state = 'change'
        current_date = loc["construction_date"] + pd.DateOffset(months=loc["time_to_first_contract"])
        end_of_life  = loc["construction_date"] + pd.DateOffset(years=max_age)        
        while current_date <= end_of_life and current_date < max_forecast_year:
            if state == 'change':
                customer = draw_customer()
            if state == 'idle':
                idle_time = np.random.poisson(mean_idle_time)
                current_date = current_date + pd.DateOffset(months=idle_time)
            else: 
                contract = create_contract(loc, current_date, initial_term_length, mean_extensions, min_sqm, mean_sqm, customer)
                contract["end_of_life"] = end_of_life
                terms = get_terms(contract)
                contract["terms"] = terms
                contract["end_date"] = terms[-1]["end"]
                contracts.append(contract)
                current_date = contract["end_date"] + pd.DateOffset(days=1)     
            state = np.random.choice(states[0], None, True, states[1])
    return contracts
        
max_forecast_year = datetime.date(2030,1,1)
contracts = pd.DataFrame.from_dict(
    create_contracts(locations, max_forecast_year, initial_term_length, mean_extensions, min_sqm, mean_sqm)
)

contracts['contract_id'] = ["A-" + str(i) for i in np.random.choice(id_range, len(contracts), False)]
terms = contracts[['contract_id', 'terms']].groupby('contract_id').terms.apply(lambda x: pd.DataFrame(x.values[0])).reset_index()
terms['term_id'] = terms.apply(lambda x: x['contract_id'] + '-' + str(x['extension']), axis=1)

#%%
sb.displot([(i['end'].to_period('M') - i['start'].to_period('M')).n for _,i in terms.iterrows()])
np.mean([(i['end'].to_period('M') - i['start'].to_period('M')).n for _,i in terms.query('extension > 0').iterrows()])
#%%
def get_effective_days(row):
    month_start = row["month"].to_timestamp("D", how="s")
    month_end = month_start + pd.DateOffset(months=1)
    days_in_month = row["month"].day
    eff_days = days_in_month
    if row["start"] > month_start:
        eff_days = eff_days - (row["start"] - month_start).days
    if row["end"] < month_end:
        eff_days = eff_days - (month_end - row["end"]).days + 1
    return pd.Series([eff_days, row["monthly_rent"]/ row["month"].day * eff_days, days_in_month], index =['effective_days', 'rent', 'daysinmonth'])

monthly_data = terms
monthly_data['month'] = monthly_data.apply(lambda x: pd.period_range(x['start'], x['end'], freq="M"), axis=1)
monthly_data  = monthly_data.explode('month')
monthly_data[['effective_days', 'rent', 'days_in_month']] = monthly_data.apply(get_effective_days, axis=1)

#%%
customers[['name', 'customer_id']].to_csv('~/coding/real-estate/customers.csv', index=False)
locations[['location_id', 'address', 'construction_date']].to_csv('~/coding/real-estate/locations.csv', index=False)
contracts[['start_date', 'end_date', 'customer_id', 'location_id', 'contract_id']].to_csv('~/coding/real-estate/contracts.csv', index=False)
terms[['contract_id', 'extension', 'start', 'end', 'sqm', 'monthly_rent', 'term_id']].to_csv('~/coding/real-estate/terms.csv', index=False)
monthly_data[['term_id', 'month', 'sqm', 'rent', 'effective_days', 'days_in_month']].to_csv('~/coding/real-estate/monthly_data.csv', index=False)

#%%
daily_data = terms    
daily_data['day'] = daily_data.apply(lambda x: pd.date_range(x['start'], x['end']), axis=1)
daily_data = daily_data.explode('day')
    
daily_data['month_end'] = (daily_data['day'] + pd.offsets.MonthEnd())
daily_data['daysinmonth'] = daily_data.month_end.apply(lambda x: x.day)
daily_data['rent'] = daily_data['monthly_rent'] / daily_data['daysinmonth']
daily_data = daily_data.drop(['month_end', 'start', 'end', 'monthly_rent', 'daysinmonth'], axis=1)
daily_data[['term_id', 'day', 'sqm', 'rent']].to_csv('~/coding/real-estate/daily_data.csv', index=False)