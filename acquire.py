import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from env import get_connection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer



def get_zillow_data():
    '''
    This function is to get the zillow dataset from a local csv file or from SQL Ace to our working notebook to be able to
    use the data and perform various tasks using the data
    ''' 
    
    if os.path.isfile('zillow.csv'):
        
        return pd.read_csv('zillow.csv')
    
    else:
       
        url = get_connection('zillow')
        
        test = '%'
        
        query = (f'''
        SELECT yearbuilt, lotsizesquarefeet,logerror, longitude, latitude,
        transactiondate, bathroomcnt, bedroomcnt, fips,
         calculatedfinishedsquarefeet,regionidzip, taxvaluedollarcnt 
        FROM properties_2017
        JOIN propertylandusetype USING(propertylandusetypeid)
        JOIN predictions_2017 USING(id) 
        WHERE propertylandusedesc = "Single Family Residential" AND predictions_2017.transactiondate LIKE "2017{test}{test}";
        ''')

        df = pd.read_sql(query, url)
        
        df.to_csv('zillow.csv', index = False)

        return df   

