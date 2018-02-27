
# coding: utf-8

# In[1]:


# Quantitative Marketing
# Assignment 1 - Avazu Kaggle competition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[40]:


# now we set up the iteration and model the entire dataset
chunk_iter = pd.read_csv("trainshuf.csv", chunksize=10 ** 5, names=['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id',                                                                       'site_domain', 'site_category', 'app_id','app_domain',                                                                       'app_category', 'device_id', 'device_ip',                                                                       'device_model', 'device_type', 'device_conn_type',                                                                       'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'])

# we are not using a prior and we implement binarization afterwards
# the label encoder in current state is a remnant of prevous versions that would be tedious to remove
nb_train = BernoulliNB(binarize=1) 
le = LabelEncoder()


# In[41]:


# to get these values we need to change the for loop
results = pd.DataFrame(np.zeros((10 * 43, 2)), columns=['current test CTR', 'predicted CTR'])


# In[21]:


#dictionaries for splitting the dataset into chunks
df = {}
for i in range(1, 46):
    df["train{0}".format(i)] = chunk_iter.get_chunk()


# In[22]:


df.keys()


# In[23]:


df['train1'].head()


# In[24]:


# after we load the chunks we need to change the variable types
    
for j in range(1, 46):
    
    d = df["train{0}".format(j)]
    d = d.drop('id', axis=1)
    d = d.drop('C1', axis=1)
    d = d.drop('C14', axis=1)
    hours = [str(k)[-2:] for k in d['hour']]
    d['hours_clean'] = hours
    d = d.drop('hour', axis=1)
    d.loc[:, 'C20'] = d.loc[:, 'C20'] + 1 # there are values -1 and those are not allowed in NB
    
    for k in d.loc[:, 'site_id':'device_model'].keys():
        le.fit(d.loc[:, k].unique())
        d.loc[:, k] = le.transform(d.loc[:, k])
    
    df["train{0}".format(j)] = d
    
    print(j)


# In[38]:


test = df['train45']
test.head()


# In[26]:


type(df['train1'])


# In[42]:


# %%timeit
# our actual calculation
for j in range(0, 10):
    for i in range(1, 44):
        g = np.random.randint(low = 1, high = 44)
        d = df["train{0}".format(g)]

        # then we fit the Naive Bayes
        fit_train = nb_train.partial_fit(X=d.iloc[:, 1:], y = d.iloc[:, 0], classes=[0, 1])
        pred_full = fit_train.predict_proba(test.iloc[:, 1:])

        results.iloc[j * 43 + (i - 1), :] = ([np.mean(test.iloc[:, 0]), np.mean(pred_full[:, 1])])
    
    print(j)


# In[43]:


print(results.head())
results.iloc[:44, 1].plot(c='blue')
results.iloc[44:88, 1].plot(c='green')
results.iloc[88:132, 1].plot(c='lightgreen')
results.iloc[132:176, 1].plot(c='lightblue')
results.iloc[176:, 1].plot(c='red')
plt.title('CTR predicted via Naive Bayes model')
plt.xlabel('iteration')
plt.ylabel('CTR')
# plt.savefig('nbCTR.png', dpi = 1000)
plt.show()


# In[30]:


print(results.iloc[:43, 1])

