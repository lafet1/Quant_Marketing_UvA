
# coding: utf-8

# In[63]:


# Quantitative Marketing
# Assignment 1 - Avazu Kaggle competition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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


# In[33]:


# trying out code for the loop
data.loc[:5, 'site_id':'device_model'].keys()

for i in data.loc[:5, 'site_id':'device_model'].keys():
    print(i, type(i))


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


# In[89]:


# now we set up the iteration and model the entire dataset
chunk_iter = pd.read_csv("trainshuf.csv", chunksize=10 ** 5, names=['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id',                                                                       'site_domain', 'site_category', 'app_id','app_domain',                                                                       'app_category', 'device_id', 'device_ip',                                                                       'device_model', 'device_type', 'device_conn_type',                                                                       'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'])

nb_train = BernoulliNB() # we are using prior that we have taken from the previous step
le = LabelEncoder()


# In[90]:


# to get these values we need to change the for loop
results = pd.DataFrame(np.zeros((5 * 43, 2)), columns=['current test CTR', 'predicted CTR'])


# In[37]:


#dictionaries for splitting the dataset into chunks
df = {}
for i in range(1, 46):
    df["train{0}".format(i)] = chunk_iter.get_chunk()


# In[38]:


df.keys()


# In[39]:


df['train1'].head()


# In[40]:


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


# In[41]:


test = df['train45']
test.head()


# In[60]:


enc = OneHotEncoder(categorical_features='all', dtype='int64',
       handle_unknown='error', n_values='auto', sparse=True)

for j in range(1, 46):
    d = df["train{0}".format(j)]
    
    for k in d.keys():
        le.fit(d.loc[:, k].unique())
        d.loc[:, k] = le.transform(d.loc[:, k])
        
    df["train{0}".format(j)] = d


# In[101]:


get_ipython().run_cell_magic('timeit', '', '# our actual calculation\nfor j in range(0, 5):\n    for i in range(1, 44):\n        g = np.random.randint(low = 1, high = 44)\n        d = df["train{0}".format(g)]\n\n        print(i)\n\n        # then we fit the Naive Bayes\n        fit_train = nb_train.partial_fit(X=d.iloc[:, 1:], y = d.iloc[:, 0], classes=[0, 1])\n        pred_full = fit_train.predict_proba(test.iloc[:, 1:])\n\n        results.iloc[j * 43 + (i - 1), :] = ([np.mean(test.iloc[:, 0]), np.mean(pred_full[:, 1])])')


# In[100]:


print(results.head())
results.iloc[:44, 1].plot(c = 'blue')
results.iloc[44:88, 1].plot(c = 'green')
results.iloc[88:132, 1].plot(c = 'lightgreen')
results.iloc[132:176, 1].plot(c = 'lightblue')
results.iloc[176:, 1].plot(c = 'red')
plt.title('CTR predicted via Naive Bayes model')
plt.xlabel('iteration')
plt.ylabel('CTR')
plt.savefig('nbCTR.png', dpi = 1000)
plt.show()


# In[96]:


print(results.iloc[:43, 1])

