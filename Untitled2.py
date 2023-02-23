#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df = pd.read_csv('C:/Users/Ehsan/Desktop/Python/Mouse_Viral_Study.csv')


# In[7]:


df.head()


# In[9]:


sns.scatterplot(data = df,
               x = 'Med_1_mL',
               y= 'Med_2_mL',
               hue = 'Virus Present')


# In[11]:


sns.scatterplot(data = df,
               x = 'Med_1_mL',
               y= 'Med_2_mL',
               hue = 'Virus Present')
x = np.linspace(0,10,100)
m = -1
b = 11
y = m*x + b
plt.plot(x,y,'k')


# In[12]:


from sklearn.svm import SVC


# In[14]:


y = df['Virus Present']
X = df.drop('Virus Present', axis=1)


# In[15]:


print(X.shape, y.shape)


# In[16]:


from sklearn.model_selection import train_test_split_split


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)


# In[18]:


print(X_train.shape)


# In[19]:


print(X_test.shape)


# In[23]:


print(y_train.shape, y_test.shape)


# In[24]:


X_train


# In[25]:


model = SVC(kernel = 'linear', C = 1000)


# In[26]:


model.fit(X_train, y_train)


# In[ ]:




