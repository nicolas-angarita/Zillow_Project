import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from env import get_connection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer


def rename_columns(df):
    
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                   'bathroomcnt':'bathrooms',
                   'calculatedfinishedsquarefeet':'sqft',
                   'yearbuilt':'year_built',
                   'longitude':'long',
                   'latitude': 'lat',
                   'transactiondate': 'transaction_month',
                   'lotsizesquarefeet': 'lot_sqft',
                   'taxvaluedollarcnt':'home_value'})
    return df

def outlier_remove(df):

    df = df[(df.bedrooms <= 6) & (df.bedrooms > 0)]
    
    df = df[(df.bathrooms <= 6) & (df.bathrooms >= 1)]

    df = df[df.home_value < 2_000_000]

    df = df[df.sqft < 10000]
    
    df = df[df.lot_sqft < 20000]

    return df
    
def clean_zillow_data(df):
    
    df = rename_columns(df)
    
    df = df.dropna() 
    
    df = outlier_remove(df)
    
    county_map = {6037: 'los angeles', 6111: 'ventura', 6059: 'orange'}
    df['county'] = df['fips'].replace(county_map)
    
    # df['transaction_month'] = df['transaction_month'].str.replace("-", "").astype(float)

    # df['transaction_month']= df['transaction_month'].astype(str).str[4:6].astype(int)
    
    df.drop_duplicates(inplace=True)
    
    # df = df.drop(columns = ['regionidzip'])

    df.to_csv("zillow.csv", index=False)

    return df    


def train_val_test(df, stratify = None):
    seed = 22
    
    ''' This function is a general function to split our data into our train, validate, and test datasets. We put in a dataframe
    and our target variable to then return us the datasets of train, validate and test.'''
    
    train, test = train_test_split(df, train_size = 0.7, random_state = seed, stratify = None)
    
    validate, test = train_test_split(train, train_size = 0.5, random_state = seed, stratify = None)
    
    return train, validate, test


def wrangle_zillow():
  
    df = get_zillow_data()

    df = clean_zillow_data(df)
    
    return df 

