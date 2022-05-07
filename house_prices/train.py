#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import house_prices.preprocess as preprocess
import joblib


# In[99]:


def build_model(features):  
    target = features['SalePrice']
    features_train, features_validation, target_train,  target_validation = preprocess.split_data(features, target)
    features_train = preprocess.model_preprocessing(features_train)
    multi_regression_model = LinearRegression()
    multi_regression_model.fit(features_train, target_train)
    joblib.dump(multi_regression_model, "../models/model")
    evaluate = model_evaluation(features_validation, target_validation)
    return evaluate


# In[100]:


def model_evaluation(features_validation, target_validation):
    features_validation = preprocess.model_preprocessing(features_validation)
    model = joblib.load("../models/model")
    feature_target_prediction = model.predict(features_validation)
    feature_target_prediction = pd.DataFrame(feature_target_prediction,
                                             columns=['SalePrice'])
    rmse = preprocess.compute_rmsle(target_validation,
                                    feature_target_prediction)
    return rmse


# In[101]:


def model_training(features_train, target_train):
    multi_regression_model = LinearRegression()
    multi_regression_model.fit(features_train, target_train)
    joblib.dump(multi_regression_model, "../models/model")


# In[ ]:




