#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df = pd.read_csv('B:\ohe_data_reduce_cat_class.csv')


# In[6]:


df.head()


# In[8]:


df.shape


# In[40]:


df=df.drop(['Unnamed: 0'], axis=1)
df.head()


# In[41]:


X = df.drop("price", axis=1)
Y = df['price']


# In[42]:


print('Shape of X = ', X.shape)
print('Shape of Y = ', Y.shape)


# In[44]:


from sklearn import model_selection
# 0.3 means 30% will be used for testing and 0.7 or 70% data will be used for training
X_Train, X_Test, Y_Train, Y_Test = model_selection.train_test_split(X, Y, test_size = 0.2, random_state = 51)


# In[45]:


print("Sample in training set...", X_Train.shape)
print("Sample in testing set...", X_Test.shape)
print("Sample in training set...", Y_Train.shape)
print("Sample in testing set...", Y_Test.shape)


# In[47]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_Train)
X_Train= sc.transform(X_Train)
X_Test = sc.transform(X_Test)


# In[53]:


from sklearn.metrics import mean_squared_error


# In[54]:


def rmse(Y_Test, Y_Pred):
  return np.sqrt(mean_squared_error(Y_Test, Y_Pred))


# In[55]:


from sklearn.svm import SVR
svr = SVR()
svr.fit(X_Train,Y_Train)
svr_score=svr.score(X_Test,Y_Test) 
svr_rmse = rmse(Y_Test, svr.predict(X_Test))
svr_score, svr_rmse


# In[62]:


from sklearn.linear_model import LinearRegression


# In[63]:


lr = LinearRegression()


# In[65]:


lr.fit(X_Train, Y_Train)
lr_score = lr.score(X_Test, Y_Test) 
lr_rmse = rmse(Y_Test, lr.predict(X_Test))
lr_score, lr_rmse


# In[67]:


import xgboost
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_Train,Y_Train)
xgb_reg_score=xgb_reg.score(X_Test,Y_Test) # with 0.8838865742273464
xgb_reg_rmse = rmse(Y_Test, xgb_reg.predict(X_Test))
xgb_reg_score, xgb_reg_rmse


# In[69]:


print(pd.DataFrame([{'Model': 'Linear Regression','Score':lr_score, "RMSE":lr_rmse},
              {'Model': 'Support Vector Machine','Score':svr_score, "RMSE":svr_rmse},
              {'Model': 'XGBoost','Score':xgb_reg_score, "RMSE":xgb_reg_rmse}],
             columns=['Model','Score','RMSE']))


# In[ ]:




