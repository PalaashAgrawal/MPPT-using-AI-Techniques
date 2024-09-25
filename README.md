# Transformer based Time Series Prediction of the Maximum Power Point for photovoltaic cells

Dataset and code used for the paper ([link](https://scijournals.onlinelibrary.wiley.com/doi/full/10.1002/ese3.1226)) 

Dataset:
The dataset used in this project is sourced from the National Renewable Energy Laboratory’s (NREL) System Advisor Model (SAM) (https://sam.nrel.gov/), an application developed by National Renewable Energy Laboratory, USA. The weather data is simulated for 50 cities in India, providing half-hourly data points. This includes many descriptiive features of ambient weather conditions, including Solar irradiance (DNI, DHI), Ambient temperature, Wind speed, Humidity levels, and  Pollution levels (PM10 concentration).

This dataset consists of half-hourly weather files for 49 different cities in USA. 

Results:

Actual optimized power produced by a 2mx2m generic PV cell:

![image_actual](actual.png)

Results from the model:

![image_pred](prediction.png)

Power prediction error: 0.47%
