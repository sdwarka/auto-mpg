#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[ ]:


# read the dataset
print('reading the dataset')
cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
        'acceleration', 'model year', 'origin', 'car name']
df = pd.read_csv('auto-mpg.data', sep='\s+', header=None, names=cols)


# In[ ]:


# convert horsepower column to float and drop rows where horsepower = nan
print('preprocessing')
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df.dropna(axis=0, how='any', inplace=True)


# In[ ]:


# segregate features and target columns
print('segregate features and target columns')
X = df.drop(['mpg', 'car name'], axis=1)
y = df['mpg']


# In[ ]:


# split data into training and test datasets
print('split data into training and test datasets')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


# create and fit a linear regression model with the training data
print('create and fit an ML model with the training data')
lr_model = LinearRegression()
lr_model = lr_model.fit(X_train, y_train)


# In[ ]:


# obtain predictions from the model
print('obtain predictions from the model')
preds = lr_model.predict(X_test)
preds = [round(x, 2) for x in preds]


# In[ ]:


# compare actual and predicted values
print('compare actual and predicted values')
comp = pd.DataFrame()
comp['actual'] = y_test
comp['predicted'] = preds
comp['err'] = abs(comp['actual'] - comp['predicted'])
comp['pct_err'] = round(comp['err'] / comp['actual'], 2)
mean_err = round(comp['pct_err'].mean(), 2)


# In[ ]:


accuracy = 1.0 - mean_err
print(f'mean error: {mean_err}, accuracy: {accuracy}')


# In[ ]:




