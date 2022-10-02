#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("C://Users//dell//Desktop//datasets//housing.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


# There are 10 columns 
# Objective - Predict the pricing of houses
# The output column(median_house_value) is continuous numerical data hence it is a regression problem


# In[6]:


# Hypothesis Testing


# In[7]:


df.isnull().sum()


# In[8]:


# dropping the null rows as it is only 1 percent of the dataset
df.dropna(inplace=True)


# In[9]:


df.ocean_proximity.value_counts()


# In[10]:


df.corr()


# In[11]:


# There is a huge correlation between median_house_income and median_income


# In[12]:


# Visual analysis pf variables
sns.barplot(x="ocean_proximity",y="median_house_value", data=df)


# In[13]:


sns.scatterplot(x="median_income",y="median_house_value", data=df)


# In[14]:


#one hot encoding
df=pd.get_dummies(df,drop_first=True)


# In[15]:


df


# In[16]:


x=df.drop("median_house_value",axis=1)


# In[17]:


y=df[["median_house_value"]]


# In[18]:


# Split the dataset into training and testing
from sklearn.model_selection import train_test_split


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)


# In[20]:


'''Linear Regression Model'''


# In[21]:


# Scale the Data using standard scaling
from sklearn.preprocessing import StandardScaler


# In[22]:


scalar=StandardScaler()
x_train_s=scalar.fit_transform(x_train)
x_test_s=scalar.transform(x_test)


# In[23]:


# Implement the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train_s,y_train)
y_pred = model.predict(x_test_s)


# In[24]:


# To check the accuracy
from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y_test,y_pred)


# In[25]:


r2_score(y_test,y_pred)


# In[27]:


# Using Voting Ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor


# In[28]:


estimators_list = [("KNN",KNeighborsRegressor(n_neighbors=7)),("LR",LinearRegression()),("DT",DecisionTreeRegressor(max_depth=4)),("RF",RandomForestRegressor(max_depth=4))]


# In[30]:


estimators_list


# In[31]:


model = VotingRegressor(estimators = estimators_list)


# In[32]:


model.fit(x_train_s,y_train)


# In[33]:


ypred_2 = model.predict(x_test_s)


# In[35]:


# Checking Accuracy
mean_squared_error(y_test,ypred_2)


# In[36]:


r2_score(y_test, ypred_2)


# In[ ]:




