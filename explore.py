import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from env import get_connection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from scipy import stats


def county_stats_test(train):
    ''' 
    This function takes in the train dataset and outputs the T-Test results for hypothesis 3
    in the zillow regression project addressing home value of homes in Orange County against
    homes in LA or Ventura.
    '''
    # Create the samples
    orange_homes = train[train.county == 'orange_county']['home_value']
    la_ventura_homes = train[(train.county == 'la_county')|(train.county == 'ventura_county')]['home_value']

    # Set alpha
    α = 0.05

    # Check for equal variances
    s, pval = stats.levene(orange_homes, la_ventura_homes)

    # Run the two-sample, one-tail T-test.
    # Use the results from checking for equal variances to set equal_var
    t, p = stats.ttest_ind(orange_homes, la_ventura_homes, equal_var=(pval > α))

    # Evaluate results based on the t-statistic and the p-value
    if p/2 < α and t < 0:
        print('''Reject the Null Hypothesis. 
            Findings suggest there is less value or equal value in Orange County homes than homes in LA or Ventura.''')
    else:
        print('''Fail to reject the Null Hypothesis. Findings suggest there is greater home values in Orange County homes than homes in LA or Ventura''')
        
    
    
    
    
# Visuals
                 
def plot_variable_pairs(df):
    sns.pairplot(data = df.sample(2000), kind='reg', diag_kind='hist')
    
    return plt.show()
                 
                 
def plot_categorical_and_continuous_vars(df, cat, cont):
    
    df_sample = df.sample(2000)
    
    sns.barplot(x=cat, y=cont, data=df_sample, color = 'dodgerblue', edgecolor = 'black', alpha = .8)
    plt.title(f'Home Values compared to {cat}')
    
    return plt.show()
                 
def outline(df):
    fig, ax = plt.subplots(figsize = (7,5))
    sns.scatterplot(data = df,x = 'long', y ='lat', zorder = 1, palette = None, hue = 'county', 
                    hue_order = ['ventura', 'orange', 'los angeles'])
    plt.title('Location of homes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()                 
                 

def lm_plot(df):
    sns.lmplot(x = 'lot_sqft' , y = 'home_value', data = df.sample(2000), line_kws={'color': 'red'})
    plt.show()