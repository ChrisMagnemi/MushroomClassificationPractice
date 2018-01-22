
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


# Create a function to compute entropy of a pandas series

# In[146]:

def computeEntropy(series):
    numrows = float(len(series))
    entropy = 0
    property_counts = series.value_counts()
    property_names = series.value_counts().index.tolist()
    if sum(property_counts) != numrows:
        return 'sums do not match up'
    elif len(property_counts) != len(property_names):
        return 'something doesnt match up'
    for prop in property_counts:
        p_i = float(prop)/numrows
        entropy += (-1)*p_i*math.log(p_i,2)
    return entropy
        
        


# In[147]:

computeEntropy(mush['ring-number'])


# Make sure this function works properly:

# In[138]:

series = mush['ring-number']
len(series)


# In[139]:

series.value_counts()


# In[141]:

p_1 = float(7488)/float(8124)
p_2 = float(600)/float(8124)
p_3 = float(36)/float(8124)
entropy = -p_1*math.log(p_1,2) -p_2*math.log(p_2,2) -p_3*math.log(p_3,2)
print "ring-number entropy is:"
print entropy


# In[ ]:



