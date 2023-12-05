#!/usr/bin/env python
# coding: utf-8

# In[26]:


# imported libraries which are used in this Project
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[27]:


# reading the data from datasets
data = pd.read_csv('Advertising.csv')


# In[29]:


data.info()


# ### Preprocess the Data

# In[30]:


# step:1 check for missing values
data.isnull().sum()


# In[31]:


# step:2 check for duplicate values
data.duplicated().sum()


# In[32]:


# Deleting the unwanted columns from the dataset
data.drop('Ad Topic Line',axis=1,inplace=True)
data.drop('City',axis=1,inplace=True)
data.drop('Country',axis=1,inplace=True)
data.drop('Timestamp',axis=1,inplace=True)


# In[33]:


data.info()


# In[34]:


# we have stored variable 'Clicked on Ad' in y variable
y = data['Clicked on Ad']


# In[35]:


# Now we drop the dependant variable from data
data.drop('Clicked on Ad',axis=1,inplace=True)
x = data


# In[36]:


#We have changed datatype from float to integer
x['Daily Time Spent on Site'] = x['Daily Time Spent on Site'].astype(int)
x['Daily Internet Usage'] = x['Daily Internet Usage'].astype(int)
x['Area Income'] = x['Area Income'].astype(int)


# In[37]:


data.info()


# ### Spliting Data Set into Train Test Split

# In[38]:


# we have imported train text split from sci-kit learn library
from sklearn.model_selection import train_test_split


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[40]:


x_train.shape


# In[41]:


y_test.shape


# ### Importing logistic Regression Algorithm

# In[42]:


# We have imported logistic Regression from Sci-kit learn library
from sklearn.linear_model import LogisticRegression


# In[43]:


lr = LogisticRegression()


# In[44]:


lr.fit(x_train,y_train)


# In[45]:


lr.score(x_test,y_test)


# ### Conclusion
# ### Logistic regression gives good results on this Data set. logistic regression score is 0.93 or 93%

# In[ ]:




