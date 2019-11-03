#
#   From https://towardsdatascience.com/machine-learning-for-anomaly-detection-and-condition-monitoring-d4614e7de770
#


# Common imports
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# %matplotlib inline

from numpy.random import seed

from keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json

# An assumption is that gear degradation occur gradually over time, so we use
#  one datapoint every 10 minutes in the following analysis. Each 10 minute 
# datapoint is aggregated by using the mean absolute value of the vibration 
# recordings over the 20.480 datapoints in each file. We then merge together 
# everything in a single dataframe.

# In the following example, I use the data from the 2nd Gear failure test 
# (see readme document for further info on that experiment).

data_dir = '2nd_test'
merged_data = pd.DataFrame()

for filename in os.listdir(data_dir):
    print(filename)
    dataset=pd.read_csv(os.path.join(data_dir, filename), sep='\t')
    dataset_mean_abs = np.array(dataset.abs().mean())
    dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
    dataset_mean_abs.index = [filename]
    merged_data = merged_data.append(dataset_mean_abs)
#%%
merged_data.columns = ['Bearing 1','Bearing 2','Bearing 3','Bearing 4']

# After loading the vibration data, we transform the index to datetime format 
# (using the following convention), and then sort the data by index in 
# chronological order before saving the merged dataset as a .csv file

merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
merged_data = merged_data.sort_index()
merged_data.to_csv('merged_dataset_BearingTest_2.csv')
print(merged_data.head())

# Before setting up the models, we need to define train/test data. To do this, 
# we perform a simple split where we train on the first part of the dataset 
# (which should represent normal operating conditions), and test on the 
# remaining parts of the dataset leading up to the bearing failure.

dataset_train = merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
dataset_test = merged_data['2004-02-13 23:52:39':]
dataset_train.plot(figsize = (12,6))
plt.show()

# I then use preprocessing tools from Scikit-learn to scale the input variables
#  of the model. The “MinMaxScaler” simply re-scales the data to be in the 
# range [0,1].

scaler = preprocessing.MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                       columns=dataset_train.columns, 
                       index=dataset_train.index
                      )
# Random shuffle training data
X_train.sample(frac=1)
X_test = pd.DataFrame(scaler.transform(dataset_test), 
                      columns=dataset_test.columns, 
                      index=dataset_test.index
                     )

# # As an initial attempt, let us compress the sensor readings down to the 
# # two main principal components.

# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, svd_solver= 'full')

# X_train_PCA = pca.fit_transform(X_train)
# X_train_PCA = pd.DataFrame(X_train_PCA)
# X_train_PCA.index = X_train.index

# X_test_PCA = pca.transform(X_test)
# X_test_PCA = pd.DataFrame(X_test_PCA)
# X_test_PCA.index = X_test.index

# Defining the Autoencoder network:
# We use a 3 layer neural network: First layer has 10 nodes, middle layer has 2
# nodes, and third layer has 10 nodes. We use the mean square error as loss 
# function, and train the model using the “Adam” optimizer.

seed(10)
#set_random_seed(10)
act_func = 'elu'

# Input layer:
model=Sequential()
# First hidden layer, connected to input vector X. 
model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(X_train.shape[1],)
               )
         )

model.add(Dense(2,activation=act_func, kernel_initializer='glorot_uniform'))
model.add(Dense(10,activation=act_func, kernel_initializer='glorot_uniform'))
model.add(Dense(X_train.shape[1], kernel_initializer='glorot_uniform'))

model.compile(loss='mse', optimizer='adam')

# Train model for 100 epochs, batch size of 10: 
NUM_EPOCHS=100
BATCH_SIZE=10

# Fitting the model:
# To keep track of the accuracy during training, we use 5% of the training data
# for validation after each epoch (validation_split = 0.05)

history = model.fit(np.array(X_train),np.array(X_train),
                  batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)

# Visualize training/validation loss:

plt.plot(history.history['loss'],
         'b',
         label='Training loss')
plt.plot(history.history['val_loss'],
         'r',
         label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0,.1])
plt.show()

# Distribution of loss function in the training set:
# By plotting the distribution of the calculated loss in the training set, 
# one can use this to identify a suitable threshold value for identifying an
# anomaly. In doing this, one can make sure that this threshold is set above
# the “noise level”, and that any flagged anomalies should be statistically 
# significant above the noise background.

X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred, columns=X_train.columns)
X_pred.index = X_train.index

scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
plt.figure()

sns.distplot(scored['Loss_mae'], bins=10, kde=True, color='blue')
plt.xlim([0.0,.5])
plt.show()

# From the above loss distribution, let us try a threshold of 0.3 for flagging 
# an anomaly. We can then calculate the loss in the test set, to check when the 
# output crosses the anomaly threshold.

X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred, columns=X_test.columns)
X_pred.index = X_test.index

scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = 0.3
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
print(scored.head())

# We then calculate the same metrics also for the training set, and merge all
#  data in a single dataframe:

X_pred_train = model.predict(np.array(X_train))
X_pred_train = pd.DataFrame(X_pred_train, columns=X_train.columns)
X_pred_train.index = X_train.index

scored_train = pd.DataFrame(index=X_train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
scored_train['Threshold'] = 0.3
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])

# Results from Autoencoder model:
# Having calculated the loss distribution and the anomaly threshold, we can
#  visualize the model output in the time leading up to the bearing failure:

scored.plot(logy=True, figsize=(10,6), ylim=[1e-2,1e2], color=['blue','red'])
plt.show()