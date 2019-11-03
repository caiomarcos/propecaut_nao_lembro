# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 22:35:34 2019

@author: caiom
"""

# Imports
import os
import pandas as pd
import numpy as np
from numpy.random import seed

import seaborn as sns
sns.set(color_codes=True)

import matplotlib.pyplot as plt

from sklearn import preprocessing

from keras import regularizers
from keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from keras.models import model_from_json

from scipy.stats import kurtosis
from scipy.stats import skew
##%%
#def calc_RMSEE(dataset):
#    for value in dataset:
#        rmsee = 1/
#%%
#Set dir where data is
data_dir = '2nd_test'
#count the number of files
file_list = os.listdir(data_dir)
number_files = len(file_list)
#dataset
df_rms = pd.DataFrame()
df_kurt = pd.DataFrame()
df_skw = pd.DataFrame()
#%%
#for each file in folder
for channel in [0, 1, 2, 3]:
    for filename in os.listdir(data_dir):
        print(filename)
        #make a dataset with column values in the file
        dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t', header=None, usecols=[channel])
        #calculate rms for those values
        rms = np.sqrt(np.mean(dataset**2))
        #make it a dataframe object
        rms = pd.DataFrame(np.array(rms).T)
        #calculate kurtosis for those values
        kurt = kurtosis(dataset)
        #make it a dataframe object
        kurt = pd.DataFrame(np.array(kurt).T)
        #calculate skewness for those values
        skw = skew(dataset)
        #make it a dataframe object
        skw = pd.DataFrame(np.array(skw).T)
        #append rms to the rms values dataset
        df_rms = df_rms.append(rms, ignore_index=True)
        #append kurtosis to the kurtosis values dataset
        df_kurt = df_kurt.append(kurt, ignore_index=True)
        #append skewness to the skewness values dataset
        df_skw = df_skw.append(skw, ignore_index=True)
    #make csv out of the calculated values
    #create dataframe with the correct size
    dataset = pd.DataFrame(index=range(number_files), columns=range(3))
    #name dataframe columns
    dataset.columns = ['RMS', 'kurtosis', 'skewness']
    #assign values to columns
    dataset['RMS'] = df_rms
    dataset['kurtosis'] = df_kurt
    dataset['skewness'] = df_skw
    #create .csv
    name = '3rd_or_4th_test_-ch'
    name = name + str(channel+1)
    name = name + '.csv'
    dataset.to_csv(name)