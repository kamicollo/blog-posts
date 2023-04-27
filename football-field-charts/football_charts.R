library(tidyverse)
library(data.table)
library(RColorBrewer)

#read in the table that contains individual contract terms with associated start/end dates
terms = fread("~/coding/real-estate-football-charts/data/terms.csv")
contracts = fread("~/coding/real-estate-football-charts/data/contracts.csv")



#read in the table that contains monthly information for each term (SQM / rent income in a month)
#add proper year/month columns to the dataframe
PL = fread("~/coding/real-estate-football-charts/data/monthly_data.csv") %>% 
  mutate(date = as.Date(strptime(paste0(month,"-01"), format="%Y-%m-%d"))) %>%
  mutate(year = year(date), month = month(date))

#calculate future value per contract (assuming we're at the end of 2020)
# filter it down to top10 contracts only
top_10_contracts = PL %>% filter(date > "2020-12-31") %>% 
  inner_join(terms, by='term_id') %>%
  inner_join(contracts, by="contract_id") %>% 
  group_by(contract_id) %>% summarize(future_value = sum(rent), .groups='drop_last') %>% 
  arrange(desc(future_value)) %>% head(10)

chart_df = 
  top_10_contracts[1,] %>% #filter to 1 contract
  inner_join(terms, by="contract_id") %>% #get terms of top 10 contracts
  mutate(width = as.numeric(difftime(end, start, units="days"))) %>% #calculate # of days of each term
  mutate(mid_point = start + width/2) %>% #calculate the mid-point of each term
  mutate(extension = as.factor(extension)) #extensions are numeric, make sure they are treated as discrete variables

ggplot(chart_df) + 
  geom_tile(aes(x=mid_point, y=contract_id, width=width, height=monthly_rent, fill=extension))

chart_df = 
  top_10_contracts %>% #all 10 contracts
  inner_join(terms, by="contract_id") %>% #get terms of top 10 contracts
  mutate(width = as.numeric(difftime(end, start, units="days"))) %>% #calculate # of days of each term
  mutate(mid_point = start + width/2) %>% #calculate the mid-point of each term
  mutate(extension = as.factor(extension)) #extensions are numeric, make sure they are treated as discrete variables

ggplot(chart_df) + 
  geom_tile(aes(x=mid_point, y=contract_id, width=width, height=monthly_rent, fill=extension))
  

chart_df = 
  top_10_contracts %>%
  inner_join(terms, by="contract_id") %>% #get terms of top 10 contracts
  mutate(width = as.numeric(difftime(end, start, units="days"))) %>% #calculate # of days of each term
  mutate(mid_point = start + width/2) %>% #calculate the mid-point of each term
  mutate(extension = as.factor(extension)) #extensions are numeric, make sure they are treated as discrete variables

ggplot(chart_df) + 
  geom_tile(aes(x=mid_point, y=contract_id, width=width, height=monthly_rent/max(monthly_rent), fill=extension)) +
  ylab("") + xlab("Year") + theme_bw() + 
  theme(panel.grid.minor.y = element_blank(), panel.grid.major.y = element_blank())
  
chart_df = 
  top_10_contracts %>%
  inner_join(terms, by="contract_id") %>% #get terms of top 10 contracts
  mutate(width = as.numeric(difftime(end, start, units="days"))) %>% #calculate # of days of each term
  mutate(mid_point = start + width/2) %>% #calculate the mid-point of each term
  mutate(extension = as.factor(extension)) %>% #extensions are numeric, make sure they are treated as discrete variables
  mutate(contract_id = reorder(as.character(contract_id), future_value, mean)) #reorder contracts

labels = chart_df %>% group_by(contract_id) %>% #group by contract
  summarize(end_date = max(end), value = max(future_value), .groups="drop_last") %>% #calculate total value and get end date
  mutate(label = paste0(round(value / 1000000, 1), "M")) #format total value

ggplot(chart_df) + 
  #create rectangle geoms
  geom_tile(aes(x=mid_point, y=contract_id, width=width, height=0.8), fill="#e8e8e8") + 
  geom_tile(aes(x=mid_point, y=contract_id, width=width, height=monthly_rent/max(monthly_rent)*0.8, fill=extension)) +
  #deal with axis labels
  ylab("") + xlab("Year") + theme_bw() + 
  #remove gridlines on Y-axis
  theme(panel.grid.minor.y = element_blank(), panel.grid.major.y = element_blank()) +
  #add labels at the end of each contract bar
  geom_label(data = labels, mapping=aes(end_date + 400, contract_id, label=label)) +
  #apply a better color palette
  scale_fill_brewer(palette = "RdYlGn")


