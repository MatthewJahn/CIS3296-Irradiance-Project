#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Keras configuration

# Set KERAS backend to Theano
import os
os.environ['KERAS_BACKEND']='theano'

# Load Keras
import keras

# Load the libraries
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD


# In[2]:


import pandas as pd
from functools import reduce


# In[3]:


# Load Data into array

nodes = []
for x in range(0, 3):
    nodes.append(pd.read_csv('https://raw.githubusercontent.com/Amitangshu/Irradiance_sensing/master/All_Data_For_EPIC_Roof/MDA300n' + str(x+150) +'.dat',
                     error_bad_lines=False, header=1))
    print("Loaded file MDA300n"+str(x+150)+".dat")


# In[4]:


# Split TimeStamp into Date and Time columns and 

numNodes = len(nodes)
for y in range(numNodes):
    nodes[y][['DATE','TIME']] = nodes[y].TMSTAMP.str.split(expand=True)
    nodes[y].TIME = nodes[y].TIME.str[0:5] + ':00'


# In[5]:


for w in range(numNodes):
    nodes[w] = nodes[w].drop('TMSTAMP', 1)
    nodes[w] = nodes[w].drop('RECNBR', 1)
    nodes[w] = nodes[w].drop('PARENT', 1)
    nodes[w] = nodes[w].drop('SOLAROCV', 1)
    nodes[w] = nodes[w].drop('VBATT', 1)
    nodes[w] = nodes[w].drop('TEMP', 1)
    nodes[w] = nodes[w].drop('DATE', 1)


# In[6]:


dataR = reduce(lambda  left,right: pd.merge(left,right,on=['IRRADIANCE','TIME'],how='outer'), nodes)


# In[7]:


dataR = dataR.groupby('TIME').mean().reset_index()


# In[8]:


dataR.sort_values(by='TIME')


# In[9]:


import sklearn
from sklearn.model_selection import train_test_split

# dataR = reduce(lambda  left,right: pd.merge(left,right,on=['IRRADIANCE','TIME'],how='outer'), nodes)
# dataR.TIME = pd.to_datetime(dataR.TIME)
Train_Set_R = dataR


# In[10]:


# Training set only consists of irradiance values ordered by timestamp
trainset_order = Train_Set_R.sort_values(by='TIME')


# In[11]:


# Remove Time, retain Irradiance only
train_R = trainset_order.iloc[:, 1:2].values


# In[12]:


train_R


# In[13]:


# Data Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))


# In[14]:


# Scale the training set
train_scaled = scaler.fit_transform(train_R)


# In[15]:


# Feature variable: Irradiance values over the previous 60 minutes
# Label (to be predicted): Irradiance of the next minute (61st min)

features_set = []
labels = []
numRec = len(train_scaled)

for i in range(60, numRec):
    features_set.append(train_scaled[i-60:i,0])
    labels.append(train_scaled[i,0])


# In[16]:


# Re-shaping the training set to numpy array before feeding the model

features_set, labels = np.array(features_set), np.array(labels)

features_set = np.reshape(features_set,(features_set.shape[0],features_set.shape[1],1))


# In[17]:


features_set.shape


# In[18]:


from keras.layers import Dense
from keras.layers import LSTM

modelM = Sequential()
modelM.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
modelM.add(LSTM(units=50, return_sequences=True))
modelM.add(Dropout(0.4))

modelM.add(LSTM(units=50, return_sequences=True))
modelM.add(Dropout(0.4))

modelM.add(LSTM(units=50))
modelM.add(Dropout(0.4))


# In[19]:


modelM.add(Dense(units = 1))
modelM.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[20]:


historyM = modelM.fit(features_set, labels, epochs = 10, batch_size = 128)


# In[21]:


import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

x = pd.to_datetime(dataR.TIME)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, dataR.IRRADIANCE, color='purple', label='Actual Irradiance: Nodes 150,151,152')
ax.set(xlabel="Hour", ylabel="irradiance",
       title="Actual Irradiance")

# Format the x axis
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.legend()
plt.title('Training set: Irradiance recorded every minute')

fig.autofmt_xdate()
plt.show()


# ### Prediction with data from Node 154

# In[22]:


test_input = pd.read_csv('https://raw.githubusercontent.com/Amitangshu/Irradiance_sensing/master/All_Data_For_EPIC_Roof/MDA300n154.dat',
                     error_bad_lines=False, header=1)
print("Loaded file MDA300n154.dat")


# In[23]:


test_input[['DATE','TIME']] = test_input.TMSTAMP.str.split(expand=True)
test_input.TIME = test_input.TIME.str[0:5] + ':00'


# In[24]:


test_input = test_input.drop('TMSTAMP', 1)
test_input = test_input.drop('RECNBR', 1)
test_input = test_input.drop('PARENT', 1)
test_input = test_input.drop('SOLAROCV', 1)
test_input = test_input.drop('VBATT', 1)
test_input = test_input.drop('TEMP', 1)
test_input = test_input.drop('DATE', 1)


# In[25]:


# Drop duplicates
# test_input.drop_duplicates(subset='TIME',keep='first',inplace=True)
# Re-order
test_input = test_input.groupby('TIME').mean().reset_index()
test_input = test_input.sort_values(by='TIME')


# In[26]:


# Testing set contains recorded irradiance values 

testset = test_input.iloc[:, 1:2].values


# In[27]:


testset


# In[28]:


# Transform testing set
testset = testset.reshape(-1,1)
testset = scaler.transform(testset)


# In[29]:


# Feed the model with irradiance values recorded every minute

test_features = []
num = len(testset)

for i in range(60, num):
    test_features.append(testset[i-60:i, 0])


# In[30]:


test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))


# In[31]:


predictions = modelM.predict(test_features)


# In[32]:


# Inverse transform the prediction because inputs are scaled

predictions = scaler.inverse_transform(predictions)


# In[33]:


predictions


# In[34]:


predictions.shape


# In[35]:


test_input.TIME[60:62]


# In[36]:


x2 = test_input.TIME[60:1441]
x2.shape


# In[37]:


x = pd.to_datetime(test_input.TIME)
x2 = pd.to_datetime(x2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, test_input.IRRADIANCE, color='red', label='Actual Irradiance: Node 154')
ax.plot(x2, predictions, color='blue', label='Predicted Irradiance')
ax.set(xlabel="Hour", ylabel="Irradiance")

# Format the x axis
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.legend()
plt.title('Prediction vs Actual Irradiance: Node 154')

fig.autofmt_xdate()
plt.show()


# In[38]:


#Load GHI
ghi = pd.read_csv('https://raw.githubusercontent.com/MatthewJahn/CIS3296-Irradiance-Project/master/GHIsolcast5min.csv',
                     error_bad_lines=False, header=0)
print("Loaded file GHIsolcast.csv")

#Reduce GHI data to relvant columns
print(ghi.columns)
ghi[['DATE','TIME']] = ghi.PeriodStart.str.split('T', expand=True)
ghi.TIME = ghi.TIME.str[0:8] #theres a T at the end otherwise
ghi = ghi.drop('Period', 1)
ghi = ghi.drop('PeriodStart', 1)
ghi = ghi.drop('PeriodEnd', 1)
ghi = ghi.drop('DATE', 1)

print(ghi.columns)
ghi.shape


# In[39]:


#Avg GHI across each 5 min interval

ghiAvg = ghi.groupby(['TIME']).mean().reset_index()

print(ghiAvg.columns)
print(ghiAvg.shape)

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #print(ghiAvg)
    


# In[40]:


#Expand data to 1min resolution to match

ghiAvg2 = ghiAvg.copy(deep=True)
#ghiAvg2.TIME = ghiAvg2.TIME.str[0:4] + np.where(ghiAvg2['TIME'].str[4]=='0', '6', '1') + ':00'

ghiAvg3 = ghiAvg.copy(deep=True)
#ghiAvg3.TIME = ghiAvg3.TIME.str[0:4] + np.where(ghiAvg3['TIME'].str[4]=='0', '7', '2') + ':00'

ghiAvg4 = ghiAvg.copy(deep=True)
#ghiAvg4.TIME = ghiAvg4.TIME.str[0:4] + np.where(ghiAvg4['TIME'].str[4]=='0', '8', '3') + ':00'

ghiAvg5 = ghiAvg.copy(deep=True)
#ghiAvg5.TIME = ghiAvg5.TIME.str[0:4] + np.where(ghiAvg5['TIME'].str[4]=='0', '9', '4') + ':00'

ghiAvg = ghiAvg.append([ghiAvg2, ghiAvg3, ghiAvg4, ghiAvg5])

ghiAvg = ghiAvg.sort_values(by=['TIME'])

print(ghiAvg.TIME)


# In[41]:


x2 = test_input.TIME[60:1441]
predictions = modelM.predict(test_features)
predictions = scaler.inverse_transform(predictions)
predictions.shape


# In[42]:


x = pd.to_datetime(test_input.TIME)
x2 = pd.to_datetime(x2)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, ghiAvg.Ghi, color='red', label='Global Horizontal Irradiance')
ax.plot(x2, predictions, color='blue', label='Predicted Irradiance')
ax.set(xlabel="Hour", ylabel="Irradiance")

# Format the x axis
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
ax.legend()
plt.title('Prediction vs GHI')

fig.autofmt_xdate()
plt.show()


# In[ ]:




