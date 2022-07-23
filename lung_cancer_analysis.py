#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython


# In[26]:


data=pd.read_csv(r'C:\Users\Vinayak\Downloads\lung_cancer.csv')
data.head()


# In[27]:


data.shape


# In[28]:


data.info()


# In[29]:


data.nunique()


# In[30]:


data.isna().sum()


# In[31]:


data.duplicated().sum()


# In[33]:


for i in data.columns:
    plt.figure(figsize=(15,6))
    sns.countplot(data[i], data = data,
    palette='hls')
    plt.xticks(rotation = 90)
    plt.show()


# In[35]:


sns.pairplot(data,hue='Result')


# In[38]:



plt.figure(figsize=(15,6))
sns.boxplot(data['Age'])
plt.xticks(rotation = 90)
plt.show()


# In[39]:


plt.figure(figsize=(15,6))
sns.boxplot(data['AreaQ'])
plt.xticks(rotation = 90)
plt.show()


# In[40]:


plt.figure(figsize=(15,6))
sns.boxplot(data['Smokes'])
plt.xticks(rotation = 90)
plt.show()


# In[45]:


data1=data.drop('Name',axis='columns')


# In[46]:


data1.shape


# In[47]:


data2=data1.drop('Surname',axis='columns')
data2.shape


# In[48]:


data2.head()


# In[50]:


for i in data2.columns:
    plt.figure(figsize=(15,6))
    sns.distplot(data2[i], color='green')
    plt.tight_layout()


# In[51]:


plt.figure(figsize = (10,8))
sns.heatmap(data2.corr(),annot=True, cbar=False, cmap='Blues', fmt='.1f')
plt.show()


# In[53]:


def minmax_norm(df):
    return (data2 - data2.min()) / ( data2.max() - data2.min())
data3= minmax_norm(data2)


# In[54]:


data3.head()


# In[55]:


x=data3.drop('Result',axis='columns')
y=data3['Result']


# In[56]:


x.shape


# In[57]:


y.shape


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[60]:


from sklearn.linear_model import LogisticRegression


# In[62]:


model1=LogisticRegression(random_state=0)
model1.fit(x_train,y_train)


# In[63]:


print("Training Accuracy :", model1.score(x_train, y_train))
print("Testing Accuracy :", model1.score(x_test, y_test))


# In[64]:


from sklearn.tree import DecisionTreeClassifier


# In[65]:


classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)


# In[66]:


classifier.fit(x_train,y_train)


# In[67]:


print("Training Accuracy :", classifier.score(x_train, y_train))
print("Testing Accuracy :", classifier.score(x_test, y_test))


# In[70]:


data3[['Smokes','Alkhol','Result']].corr()


# In[ ]:




