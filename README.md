# Energy-Demand-Forecasting

## Title: Energy Demand Forecasting
This project investigates various models to predict energy demands for various datasets.

## Project Definition and Motivation:
Time series analysis comprises methods for analyzing time series data to extract meaningful statistics and other characteristics of the data. Forecasting is One of the study areas with the greatest potential to increase the amount of renewable energy in the grid. Demand forecasting is essential in managing the supply-demand of energy. Therefore, it is crucial for the power generation and distribution systems to use models to accurately predict future energy demand trends.

## Objective:
The specific issue being dealt with is how to use past energy consumption data, week days, holidays and weather data to anticipate the future energy demand. Various models are tried on different datasets and comparison is made between them.

## Technologies:
Python
Keras, Tensorflow
Pandas, Numpy, Jupyter
Many other libraries 

## Project Description
On a practical job of forecasting short-term energy consumption, this experiment evaluated the forecasting skills of traditional statistical models and contemporary neural network implementations. Models are evaluated by calculating RMSE for each experiment performed.
Models used:
1.	LSTM - Long-Short Term Memory Neural Network
2.	ARIMA - Autoregressive Integrated Moving Average
3.	Prophet by Facebook

## Datasets used:
We have explored and tried the following datasets to check the model’s efficiency:
1.	Spain’s energy and Weather dataset
2.	Electricity demand in Victoria
3.	PJM electricity consumption
4.	Electricity Demand of Ontario
5.	Electric Power Consumption of households in India

## Evaluation
RMSE which is a standard way to measure the error of a model in predicting quantitative data is used to evaluate each model. Also, MAPE and MAE are used for the forecasting. Prophet got the least RMSE which indicates it is better in making forecasting.

## Results:
After implementing the models, we concluded that Prophet outperforms the other two, followed by ARIMA and LSTM.
