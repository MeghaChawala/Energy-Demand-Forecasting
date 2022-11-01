#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima                        
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv(r'C:\Users\jeetp\data\02-compiled\demand\demand.csv')
df.head()


# In[2]:


df.info()


# In[3]:


df.Month = pd.to_datetime(df.datetime)
df = df.set_index("datetime")
df.head()


# In[4]:


df.index.freq = 'MS'
ax = df['ont_demand'].plot(figsize = (16,5), title = "Ontario Energy Demand")
ax.set(xlabel='Dates', ylabel='Total Consumption');


# In[6]:


train_data = df[:len(df)-12]
test_data = df[len(df)-12:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)


# In[7]:


from keras.preprocessing.sequence import TimeseriesGenerator

n_input = 12
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()


# In[8]:


lstm_model.fit_generator(generator,epochs=20)


# In[10]:


import matplotlib.pyplot as plt
losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);


# In[11]:


lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)


# In[12]:


lstm_predictions_scaled


# In[13]:


lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
lstm_predictions


# In[14]:


test_data['LSTM_Predictions'] = lstm_predictions
test_data


# In[15]:


test_data['ont_demand'].plot(figsize = (16,5), legend=True)
test_data['LSTM_Predictions'].plot(legend = True);


# In[24]:


lstm_rmse_error = rmse(test_data['ont_demand'], test_data["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = df['ont_demand'].mean()

print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error*10}\nMean: {mean_value}')


# In[ ]:




