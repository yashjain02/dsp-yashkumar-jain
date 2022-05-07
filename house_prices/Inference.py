#!/usr/bin/env python
# coding: utf-8

# In[23]:


import house_prices.preprocess as preprocess
import joblib


# In[24]:


def make_predictions(test):
    test_numerical = test[['OverallQual', 'GrLivArea', 'GarageCars', 
                           'GarageArea', 'TotalBsmtSF']]
    test_categorical = preprocess.get_categorical_feature(test)
    test_categorical = preprocess.encoder(test_categorical)
    test = preprocess.concat_num_cat_features(test_categorical, test_numerical)
    test = test[['OverallQual', 'GrLivArea', 'Foundation', 'Neighborhood']]
    test = preprocess.scaling(test)
    test = preprocess.fill_missing_value(test)
    model = joblib.load("../models/model")
    Final_prediction = model.predict(test)
    return Final_prediction


# In[ ]:




