
# coding: utf-8

# In[1]:


# Quantitative Marketing
# Assignment 1 - Avazu Kaggle competition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[2]:


# here we try to get a feel of the data set and model the first piece of it
data = pd.read_csv("train.csv", nrows=100000)


# In[3]:


data.describe()


# In[4]:


data.info()


# In[5]:


# data.iloc[:10, 4:]

le = LabelEncoder()

for i in range(5, 14):
    le.fit(data.iloc[:, i].unique())
    data.iloc[:, i] = le.transform(data.iloc[:, i])

data.loc[:, 'C20'] = data.loc[:, 'C20'] + 1

data.info()


# In[6]:


print(data.head())


# In[7]:


print(data.hour.unique())


# In[8]:


data.loc[:10, 'site_id']


# In[9]:


train, test = train_test_split(data, test_size=0.2)


# In[10]:


print(train.shape)
print(test.shape)


# In[11]:


model = GaussianNB(priors=[0.9, 0.1])
fit = model.fit(X=train.iloc[:, 2:], y=train.iloc[:, 1])


# In[12]:


fit.get_params()


# In[13]:


pred = fit.predict_proba(test.iloc[:, 2:])
print(pred[:10])
print(pred.shape)


# In[14]:


print(np.mean(train.iloc[:, 1]))
print(np.mean(test.iloc[:, 1]))
print(np.mean(pred[:, 1] > 0.5))
print(np.mean(pred[:, 1]))


# In[15]:


# now we set up the iteration and model the entire dataset
chunk_iter = pd.read_csv("train.csv", chunksize=10 ** 6)
results = pd.DataFrame(np.zeros((39, 3)), columns=['current test CTR', 'share of observations labelled 1', 'predicted CTR'])
full_test = pd.DataFrame(np.zeros((2 * 10 ** 6, 24)).astype(int), columns=data.columns)

nb_train = GaussianNB(priors=[0.9, 0.1]) # we are using prior that we have taken from the previous step
le = LabelEncoder()


# In[ ]:


# our actual calculation
for chunk_number, d in enumerate(chunk_iter, start=1):
    
    if chunk_number < 3:
        
        print(chunk_number)
       
        for i in range(5, 14):
            le.fit(d.iloc[:, i].unique())
            d.iloc[:, i] = le.transform(d.iloc[:, i])
        
        full_test.iloc[((chunk_number - 1) * 10 ** 6) : (chunk_number * 10 ** 6), :] = d
        
   
    else:
        
        print(chunk_number)
        # after we load the chunks we need to change the variable types
        for i in range(5, 14):
            le.fit(d.iloc[:, i].unique())
            d.iloc[:, i] = le.transform(d.iloc[:, i])

        d.loc[:, 'C20'] = d.loc[:, 'C20'] + 1 # there are values -1 and those are not allowed in NB

        data_train, data_test = train_test_split(d, test_size=0.2)

        # then we fit the Naive Bayes
        fit_train = nb_train.partial_fit(X=data_train.iloc[:, 2:], y=data_train.iloc[:, 1], classes=[0, 1])
        pred_full = fit_train.predict_proba(full_test.iloc[:, 2:])

        results.iloc[chunk_number - 3, :] = ([np.mean(full_test.iloc[:, 1]), np.mean(pred_full[:, 1] > 0.5), np.mean(pred_full[:, 1])])
    


# In[361]:


print(full_test.info())
print(full_test.hour.unique())


# In[363]:


print(d.hour.unique())


# In[357]:


print(full_test.shape)
print(full_test.tail())
print(full_test.iloc[999995:10 ** 6 + 1, :])


# In[359]:


print(results.head())
results['predicted CTR'].plot(kind='line')
plt.show()


# In[271]:


np.mean(np.logical_and(results['predicted CTR'] > 0.15, results['predicted CTR'] < 0.2))

