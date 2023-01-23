import pandas as pd
import numpy as np

import acquire as a
import prepare as p
import explore as e

import seaborn as sns
import matplotlib.pyplot as plt 


from scipy import stats
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import f_regression, SelectKBest, RFE 
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt 

import warnings
warnings.filterwarnings("ignore")



def mvp_scaled_data(train, 
               validate, 
               test, 
               columns_to_scale=['long','lat', 'logerror', 'sqft', 'lot_sqft'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    mms = MinMaxScaler()
    #     fit the thing
    mms.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(mms.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(mms.transform(validate[columns_to_scale]), 
                                                     columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mms.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

    
    
def select_kbest(x,y,k):
    
    f_selector = SelectKBest(f_regression, k = k)
    
    f_selector.fit(x, y)   
    
    f_select_mask = f_selector.get_support()

    f_selector.transform(x)
    
    
    return x.iloc[:,f_select_mask]    
    

    
def rfe(x, y, k):
    
    lm = LinearRegression()

    rfe = RFE(lm, n_features_to_select = k)
    
    rfe.fit(x, y)
    
    ranks = rfe.ranking_

    columns = x.columns.tolist()
    
    feature_ranks = pd.DataFrame({'ranking': ranks,
                                 'feature': columns})
    
    return feature_ranks.sort_values('ranking').reset_index().drop(columns = ('index'))  


def splitting_subsets(train, train_scaled, validate_scaled, test_scaled):
    
    X_train = train_scaled.drop(columns = ['year_built', 'home_value', 'fips'])
    X_train = pd.get_dummies(X_train, columns = ['transaction_month', 'bathrooms', 'bedrooms', 'county'])
    y_train = train['home_value']


    X_validate = validate_scaled.drop(columns = ['year_built', 'home_value', 'fips'])
    X_validate = pd.get_dummies(X_validate, columns = ['transaction_month', 'bathrooms', 'bedrooms', 'county'])
    y_validate = validate_scaled['home_value']


    X_test = test_scaled.drop(columns = ['year_built', 'home_value', 'fips'])
    X_test = pd.get_dummies(X_test, columns = ['transaction_month', 'bathrooms', 'bedrooms', 'county'])
    y_test = test_scaled['home_value']

    return X_train, y_train, X_validate, y_validate, X_test, y_test

    
def lasso_lars(X_train, y_train):
    metrics = []

    for i in np.arange(0.05, 1, .05):
    
        lasso = LassoLars(alpha = i )
    
        lasso.fit(X_train, y_train)
    
        lasso_preds = lasso.predict(X_train)
        
        preds_df = pd.DataFrame({'actual': y_train})
    
        preds_df['lasso_preds'] = lasso_preds

        lasso_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['lasso_preds']))
    
        output = {
                'alpha': i,
                'lasso_rmse': lasso_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('lasso_rmse')


def tweedie_models(X_train, y_train):
    metrics = []

    for i in range(0, 4, 1):
    
        tweedie = TweedieRegressor(power = i)
    
        tweedie.fit(X_train, y_train)
    
        tweedie_preds = tweedie.predict(X_train)
    
        preds_df['tweedie_preds'] = tweedie_preds
    
        tweedie_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.tweedie_preds))
    
        output = {
                'power': i,
                'tweedie_rmse': tweedie_rmse
                 }
    
        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('tweedie_rmse') 

def linear_poly(X_train, y_train):
    metrics = []

    for i in range(2,4):

        pf = PolynomialFeatures(degree = i)

        pf.fit(X_train, y_train)

        X_polynomial = pf.transform(X_train)

        lm2 = LinearRegression()

        lm2.fit(X_polynomial, y_train)
        
        preds_df = pd.DataFrame({'actual': y_train})

        preds_df['poly_preds'] = lm2.predict(X_polynomial)

        poly_rmse = sqrt(mean_squared_error(preds_df['actual'], preds_df['poly_preds']))

        output = {
                'degree': i,
                'poly_rmse': poly_rmse
                 }

        metrics.append(output)

    df = pd.DataFrame(metrics)    
    return df.sort_values('poly_rmse')   

    
def linear_model(X_train, y_train):
    
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_preds = lm.predict(X_train)
    
    preds_df = pd.DataFrame({'actual': y_train,'lm_preds': lm_preds})
    
    lm_rmse = sqrt(mean_squared_error(preds_df['lm_preds'], preds_df['actual']))
    
    df = pd.DataFrame({'model': 'linear', 'linear_rmse': lm_rmse}, index=['0']) 
                      
    return df
    
def baseline(y_train):
    
    preds_df = pd.DataFrame({'actual': y_train})
    
    preds_df['baseline'] = y_train.mean()
    
    baseline_rmse = sqrt(mean_squared_error(preds_df.actual, preds_df.baseline))

    return baseline_rmse    

def validate_models(X_train, y_train, X_validate, y_validate):
   
    lm = LinearRegression()

    lm.fit(X_train, y_train)
    
    lm_val = lm.predict(X_validate)
    
    val_preds_df = pd.DataFrame({'actual_val': y_validate})
    
    val_preds_df['lm_preds'] = lm_val

    lm_rmse_val = sqrt(mean_squared_error(val_preds_df['actual_val'], val_preds_df['lm_preds']))

    #lassolars model
    lasso = LassoLars(alpha = 0.05)

    lasso.fit(X_train, y_train)

    lasso_val = lasso.predict(X_validate)

    val_preds_df['lasso_preds'] = lasso_val

    lasso_rmse_val = sqrt(mean_squared_error(val_preds_df['actual_val'],val_preds_df['lasso_preds']))
    
    #polynomial model
    
    pf = PolynomialFeatures(degree = 3)
    
    pf.fit(X_train, y_train)
    
    X_train = pf.transform(X_train)
    X_validate = pf.transform(X_validate)
    
    lm2 = LinearRegression()
    
    lm2.fit(X_train, y_train)
    
    val_preds_df['poly_vals'] = lm2.predict(X_validate)
    
    poly_validate_rmse = sqrt(mean_squared_error(val_preds_df.actual_val, val_preds_df['poly_vals']))

    return lm_rmse_val, lasso_rmse_val, poly_validate_rmse
 
    
def test_model(X_train, y_train, X_test, y_test):
    
    pf = PolynomialFeatures(degree = 3)

    pf.fit(X_train, y_train)
    X_train = pf.transform(X_train)

    X_test = pf.transform(X_test)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    test_preds_df = pd.DataFrame({'actual_test': y_test})

    test_preds_df['poly_test'] = lm.predict(X_test)

    poly_test_rmse = sqrt(mean_squared_error(test_preds_df.actual_test, test_preds_df['poly_test']))
    
    return poly_test_rmse
    
    
def best_models(X_train, y_train, X_validate, y_validate):
    
    lm_rmse = linear_model(X_train, y_train).iloc[0,1]
    
    lasso_rmse = lasso_lars(X_train, y_train).iloc[0,1]
        
    poly_rmse = linear_poly(X_train, y_train).iloc[0,1]
    
    baseline_rmse = baseline(y_train)
    
    lm_rmse_val, lasso_rmse_val, poly_validate_rmse = validate_models(X_train, y_train, X_validate, y_validate)
    
    df = pd.DataFrame({'model': ['linear', 'lasso', 'linear_poly', 'baseline'],
                      'train_rmse': [lm_rmse, lasso_rmse, poly_rmse, baseline_rmse],
                      'validate_rmse': [lm_rmse_val, lasso_rmse_val, poly_validate_rmse, baseline_rmse]})
    
    df['difference'] = df['train_rmse'] - df['validate_rmse']
    
    return df.sort_values('train_rmse').reset_index().drop(columns = ('index'))

def best_model(X_train, y_train, X_validate, y_validate, X_test, y_test):
    
    df = best_models(X_train, y_train, X_validate, y_validate).head(1)
    
    df['test_rmse'] = test_model(X_train, y_train, X_test, y_test)
    
    df = df.drop(columns = ['difference'])

    return df


    
    
    