# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

# Your solution code here

def select_from_model(data):
    X = data.drop('SalePrice',axis=1)
    y = data['SalePrice']
    rf_model = RandomForestClassifier()
    
    select_fm = SelectFromModel(rf_model)
    select_fm.fit_transform(X,y)
    
    #print (list(X.columns[select_fm.get_support()]))
    
    return list(X.columns[select_fm.get_support()])

select_from_model(data)



