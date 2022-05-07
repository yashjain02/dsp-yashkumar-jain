#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
import joblib


# In[15]:


def numerical_features(df):
    numerical_data = df[['OverallQual', 'GrLivArea',
                         'GarageCars', 'GarageArea',
                         'TotalBsmtSF', 'SalePrice']] 
    return numerical_data


# In[16]:


def get_categorical_feature(df):
    features_categorical = df.select_dtypes(include=['object'])
    return features_categorical


# In[17]:


def concat_num_cat_features(categorical, numerical):
    categorical.index = numerical.index
    train = pd.concat([categorical, numerical], axis=1)
    return train


# In[18]:


def split_data(features, target):
    feature_train, feature_val, target_train, target_val = train_test_split(features, target, test_size=0.2, random_state=0)
    return feature_train, feature_val, target_train, target_val


# In[19]:


def Add_encoder(features_train_categorical):
    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value',
                             unknown_value=100)
    ordinal.fit(features_train_categorical)
    joblib.dump(ordinal, "../models/ordinalEncoding")


# In[20]:


def Add_Scaling_object(features_train):
    scale = StandardScaler()
    scale.fit(features_train)
    joblib.dump(scale, "../models/Scaler")


# In[21]:


def scaling(features_train):
    scaling = joblib.load("../models/Scaler")
    features_train = scaling.transform(features_train)
    return features_train


# In[22]:


def fill_missing_value_object(features_train):
    imputer = SimpleImputer(missing_values=np.nan,
                            strategy='median')
    imputer.fit(features_train)
    joblib.dump(imputer, "../models/SimpleImputer")


# In[23]:


def fill_missing_value(features_train):
    SI=joblib.load("../models/SimpleImputer")
    features_train = SI.transform(features_train)
    return features_train


# In[24]:


def encoder(features_train_categorical):
    ordinalEncoder = joblib.load("../models/ordinalEncoding")
    features_train_categorical[features_train_categorical.columns] = pd.DataFrame(ordinalEncoder.transform(features_train_categorical))
    return features_train_categorical


# In[25]:


def model_preprocessing(features_train):
    features_train_numerical = numerical_features(features_train)
    features_train_categorical = get_categorical_feature(features_train)
    Add_encoder(features_train_categorical)
    features_train_categorical = encoder(features_train_categorical)
    features_train = concat_num_cat_features(features_train_categorical,
                                             features_train_numerical)
    features_train = features_train[['OverallQual', 'GrLivArea',
                                   'Foundation', 'Neighborhood']]
    Add_Scaling_object(features_train)
    features_train = scaling(features_train)
    fill_missing_value_object(features_train)
    features_train = fill_missing_value(features_train)
    return features_train


# In[26]:


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int=2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)

