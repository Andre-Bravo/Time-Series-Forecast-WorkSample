#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import itertools
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
import statsmodels.api as sm
import pandas as pd
import numpy as np

plt.rc('axes', labelsize = 14)
plt.rc('xtick', labelsize = 12)
plt.rc('ytick', labelsize = 12)
plt.rc('text', color = 'k')


# In[2]:


sales_df = pd.read_excel('Superstore.xls')


# ## Analyze and Forecast Furniture Sales

# In[3]:


furniture_sales = sales_df[sales_df['Category'] == 'Furniture']


# In[4]:


print('Data avail for',
      furniture_sales['Order Date'].min(),
      'to',
      furniture_sales['Order Date'].max())


# ##### Preprocessing

# In[5]:


furniture_sales.columns


# In[6]:


cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode',
       'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
       'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
       'Product Name', 'Quantity', 'Discount', 'Profit']
furniture_sales.drop(cols, axis=1, inplace=True)
furniture_sales = furniture_sales.sort_values('Order Date')


# In[7]:


furniture_sales.isnull().sum()


# In[8]:


furniture_sales = furniture_sales.groupby('Order Date')['Sales'].sum().reset_index()


# In[9]:


furniture_sales.set_index('Order Date', inplace=True)


# ##### Look at mean monthly furniture sales

# In[10]:


y  = furniture_sales['Sales'].resample('MS').mean()


# In[11]:


y['2017':]


# ##### Visualization

# In[12]:


y.plot(figsize=(15, 6))
plt.show()


# Note seasonality

# ##### Time Series Decomposition

# In[13]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y,
                                         model='additive')
fig = decomposition.plot()
plt.show()


# Residuals and trend-cycle point to unstable sales

# ### Forecast Using ARIMA

# ARIMA$(p, d, q)$
# <br> Seasonality, trend, noise

# Use grid search to test and find optimal parameters for model

# In[14]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))

#Seasonal ARIMA characterized by additional term, m: Number of time steps in a single seasonal period.
#As we are using monthly data, m=12
seasonal_pdq = [(x[0],
                 x[1],
                 x[2],
                 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[15]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, 
                                                 param_seasonal, 
                                                 results.aic))
        except:
            continue


# $AIC$ value for $SARIMAX(1, 1, 1)x(1, 1, 0, 12)$ is lowest, should use

# #### Fit ARIMA

# In[16]:


mod = sm.tsa.statespace.SARIMAX(y,
                               order=(1,1,1),
                               seasonal_order=(1,1,0,12),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[17]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# Daignostics tell us we have a decent fit given:
# 1. No obvious pattern in residulas
# 2. KDE curve close to normally distributed
# 3. Q-Q Plot points lie on 45 degree line for the most part
# 4. Correlations for lag greater than one not significant

# ### Forecast Validation

# In[18]:


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'),
                              dynamic=False)
pred_ci = pred.conf_int()

#plot observed
ax = y['2014':].plot(label='observed')
#plot predicted
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
#plot CI
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()


# In[19]:


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

mse = ((y_forecasted - y_truth)**2).mean()
print('The Mean Squared Error (MSE) of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error (RMSE) of our forecasts is {}'.format(round(np.sqrt(mse), 2)))


# In[20]:


print('The RMSE of {} can be comparted to the monthly sales min of {} and max of {}'.format(round(np.sqrt(mse), 2),
                                                                                            round(y.min(), 2), 
                                                                                            round(y.max(), 2))
     )

