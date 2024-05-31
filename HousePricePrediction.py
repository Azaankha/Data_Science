#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('C:/internshala certificates/HousePricePrediction.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.drop(columns=['MSZoning','BldgType','Exterior1st','LotConfig'],inplace=True)


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.duplicated().sum()


# In[11]:


df= df.fillna(df.mean())


# In[12]:


df.info()


# In[13]:


df=pd.get_dummies(df)


# In[14]:


df.head()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[16]:


X=df.drop('SalePrice', axis=1)
y=df['SalePrice']


# In[17]:


X_train, X_test, y_train,  y_test= train_test_split(X, y, test_size=0.2, random_state=42)


# In[18]:


lr=LinearRegression()
lr.fit(X_train,y_train)


# In[19]:


y_test=lr.predict(X_test)
y_train=lr.predict(X_train)


# In[20]:


lr.predict(X)


# In[21]:


sns.jointplot(x=df['LotArea'],y=df['SalePrice'],data=df,kind='reg')


# In[ ]:




