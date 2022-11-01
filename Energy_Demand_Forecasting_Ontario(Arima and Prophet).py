#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pathlib
from datetime import datetime
import math
import sys

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('grayscale')
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error

sys.path.append("..")
from prophet import Prophet
from src.models.models import SetTempAsPower, SK_Prophet

get_ipython().run_line_magic('matplotlib', 'inline')

PROJECT_DIR = pathlib.Path.cwd().parent.resolve()
CLEAN_DATA_DIR = r'C:\Users\jeetp\OneDrive\Desktop\data\data\05-clean'

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[37]:


import pathlib
import datetime
from os import PathLike
from typing import Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


get_ipython().run_line_magic('matplotlib', 'inline')

PROJECT_DIR = pathlib.Path.cwd().parent.resolve()
RAW_DATA_DIR = PROJECT_DIR / 'data' / '01-raw' /'demand'
COMPILED_DATA_DIR = PROJECT_DIR / 'data' / '02-compiled' / 'demand'


# In[38]:


df = pd.read_csv(CLEAN_DATA_DIR + '\clean-cut.csv', parse_dates=True, index_col=0)
df = df.loc['1994': '1995']
df = df.resample('D').max()
# Just select a reasonable subset of data to test the model wrappers
df = df[['temp', 'dew_point_temp', 'week_of_year', 'daily_peak']]
df.rename(columns={'temp': 'temp_max'}, inplace=True)

y = df.pop('daily_peak')
X = df

X.head()


# In[39]:


y.tail()


# In[40]:


def bound_precision(y_actual: pd.Series, y_predicted: pd.Series, n_to_check=5):
    """
    Accepts two pandas series, and an integer n_to_check
    Series are:
    + actual values
    + predicted values
    Sorts each series by value from high to low, and cuts off each series at n_to_check
    Determines how many hits - ie how many of the indices in the actual series are in the predicted series indices
    Returns number of hits divided by n_to_check    
    """
    y_act = y_actual.copy(deep=True)
    y_pred = y_predicted.copy(deep=True)
    y_act.reset_index(drop=True, inplace=True)
    y_pred.reset_index(drop=True, inplace=True)

    act_dates =set( y_act.sort_values(ascending=False).head(n_to_check).index)
    pred_dates = set(y_pred.sort_values(ascending=False).head(n_to_check).index)
    bound_precision =  len(act_dates.intersection(pred_dates))/ n_to_check
    return bound_precision

y_act = pd.Series([ 11,12,13,14, 15,16,17, 11, 12], index = pd.date_range(start='2019-01-01', periods=9))
y_pred = pd.Series([18,11,13,14, 16,15,15, 14, 11], index = pd.date_range(start='2019-03-20', periods=9))
b_prec = bound_precision(y_act, y_pred, n_to_check=3)
b_prec


# In[41]:


X_m = X.copy(deep=True)
y_m = y.copy(deep=True)
X_train = X_m['1994'] ; y_train = y_m['1994']
X_test = X_m['1995'] ; y_test = y_m['1995']

set_temp_as_power = SetTempAsPower(col='temp_max')
set_temp_as_power.fit(X_train, y_train)
preds = set_temp_as_power.predict(X_test)
print(preds)
print()
print(mean_absolute_error(y_test, preds))
print()
print(bound_precision(y_test, preds))


# In[42]:


#prophet model
sk_prophet = SK_Prophet(regressors={'temp_max':()})
sk_prophet.fit(X_train, y_train)
preds = sk_prophet.predict(X_test)
print(preds)
print()
print(bound_precision(y_test, preds))


# In[43]:


def forecast_accuracy(y_test, preds):
    mape = np.mean(np.abs(preds - y_test)/np.abs(y_test))  # MAPE
    me = np.mean(preds - y_test)             # ME
    mae = np.mean(np.abs(preds - y_test))    # MAE
    mpe = np.mean((preds - y_test)/preds)   # MPE
    rmse = np.mean((preds - y_test)**2)**.5  # RMSE
    
                 # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            })

ans = forecast_accuracy(preds, y_test)   #fc is predicted, give predicted output of prophet instead of fc


# In[55]:


print("Forecast accuracy of Prophet Model")
ans


# In[45]:


#arima model
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


# In[46]:


arima_model = auto_arima(y_train)


# In[47]:


arima_model.summary()


# In[48]:


arima_model2 = SARIMAX(y_train, order = (2,1,1), seasonal_order = (4,0,3,12))


# In[49]:


arima_result = arima_model2.fit()


# In[50]:


arima_result.summary()


# In[51]:


fc = arima_result.forecast(365, alpha=0.05)


# In[52]:


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean(((forecast - actual)**2)**.5)*2*2  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            'corr':corr, 'minmax':minmax})

ans = forecast_accuracy(fc, y_test)   #fc is predicted, give predicted output of prophet instead of fc


# In[54]:


print("Forecast accuracy of Arima Model")
ans

