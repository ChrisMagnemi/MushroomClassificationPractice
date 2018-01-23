
# coding: utf-8

# # Task 1
# Basic imports and read in the dataset

# In[57]:

import numpy as np
import pandas as pd
import math


# In[9]:

# import the mushroom dataset
mush = pd.read_csv('mushrooms.csv')


# In[142]:

mushroom


# # Task 2
# Gather some basic sense of the data

# In[15]:

mush.dtypes


# In[16]:

mush.describe()


# In[19]:

mush.isnull().sum()


# # Task 3  
# Create functions for Entropy and Information Gain calculations.
# These will then be used to determine the best splits for a tree-based classification model

# In[237]:

def computeEntropy(series):
    numrows = float(len(series))
    entropy = 0
    property_counts = series.value_counts()
    property_names = series.value_counts().index.tolist()
    if sum(property_counts) != numrows:
        return 'sums do not match up'
    elif len(property_counts) != len(property_names):
        return 'something doesnt match up'
    k = len(property_counts)
    for prop in property_counts:
        p_i = float(prop)/numrows
        if p_i != 1:
#         print p_i
            entropy -= p_i*(math.log(p_i,k))
    return entropy
        
        


# In[238]:

computeEntropy(mush['ring-number'])


# Make sure this function works properly:

# In[164]:

series = mush['cap-shape']
len(series)


# In[165]:

series.value_counts()


# In[200]:

p_1 = float(3656)/float(8124)
p_2 = float(3152)/float(8124)
p_3 = float(828)/float(8124)
p_4 = float(452)/float(8124)
p_5 = float(32)/float(8124)
p_6 = float(4)/float(8124)
entropy = (p_1*math.log(p_1,6))+(p_2*math.log(p_2,6))+(p_3*math.log(p_3,6))+(p_4*math.log(p_4,6))
+(p_5*math.log(p_5,6)) +(p_6*math.log(p_6,6))
print "cap-shape entropy is:"
print -entropy
print p_1+p_2+p_3+p_4+p_5+p_6


# In[219]:

mush['ring-number'].value_counts()


# In[241]:

def informationGain(dataset, split_feature, target):
    cols = []
    for col in dataset.columns:
        if col != target:
            cols.append(col)
    if split_feature not in cols:
        return 'the feature to be split on is not in the feature set'
    entropy_parent = computeEntropy(dataset[target])
    prop_counts = dataset[split_feature].value_counts()
    prop_names = dataset[split_feature].value_counts().index.tolist()
    numrows = float(len(dataset[split_feature]))
    for i in range(len(prop_names)):
        df = dataset.loc[dataset[split_feature] == prop_names[i]]
        part = computeEntropy(df[target])
        part = part * (float(df.shape[0])/numrows)
        entropy_parent -= part
    return entropy_parent
    
    
    


# In[242]:

informationGain(mush, 'ring-number', 'class')


# Great, now lets loop through all columns and see which gives the best information gain. All IG's are between 0 and 1 with odor as the greatest, as expected.

# In[243]:

for col in mush.columns:
    if col != 'class':
        print col, informationGain(mush, col, 'class')


# In[181]:

mush.head()


# In[184]:

mush.loc[mush['cap-shape'] == 'x']


# In[ ]:



