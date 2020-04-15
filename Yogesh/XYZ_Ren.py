#!/usr/bin/env python
# coding: utf-8

# In[530]:


import pandas as pd
import numpy as np


# In[531]:


housing =  pd.read_csv('data.csv')


# In[532]:


housing.head()


# In[533]:


housing.info()


# In[534]:


housing.isna().sum()


# In[535]:


# Scikit-Learn provides a handy class to take care of missing values: Imputer.
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")

housing_num = housing


# In[536]:


imputer.fit(housing_num)


# In[537]:


# The imputer has simply computed the median of each attribute and stored the result
# in its statistics_ instance variable

imputer.statistics_


# In[538]:


housing_num.median().values


# In[539]:


# Now you can use this “trained” imputer to transform the training set by replacing
# missing values by the learned medians:
X = imputer.transform(housing_num)


# In[540]:


# If you want to put it back into a Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[541]:


housing_tr.head()


# In[542]:


housing_tr.isnull().sum()  # Now there is no null value in dataset.


# In[543]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
housing_tr.hist(bins=50, figsize=(20,15))
plt.show()


# In[544]:


corr_housing = housing_tr.corr()
corr_housing['MEDV'].sort_values(ascending=False)


# In[545]:


# RM         0.695668 is releated to MEDV value.


# In[546]:


# split the data set 

# random sampling methods

# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[547]:


housing_tr.info()


# In[554]:


# To avoid sampling bias use stratified sampling:
# Do stratified sampling using RM values.


# It is important to have a sufficient number of instances in your dataset for each stratum,
# or else the estimate of the stratum’s importance may be biased.
# This means that you should not have too many strata, 
# and each stratum should be large enough.

housing_tr["income_cat"] = np.ceil(housing_tr["RM"] / 1.5) # 1.5 (to limit the number of income categories)
housing_tr["income_cat"].where(housing_tr["income_cat"] < 6, 6.0, inplace=True) # merging all the categories greater than 6 into category 6.


# In[549]:


housing_tr['income_cat'].value_counts() / len(housing_tr)


# In[550]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_tr, housing_tr['income_cat']):
    strat_train_set = housing_tr.loc[train_index]
    strat_test_set = housing_tr.loc[test_index]


# In[551]:


# remove the income_cat attribute so the data is back to its original state:
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# In[ ]:




