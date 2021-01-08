#%%

import pandas as pd
import numpy as np
import datetime
import re
import json
import csv
from io import StringIO
from bs4 import BeautifulSoup
import requests
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
from dateutil.relativedelta import relativedelta
import pandas_datareader as web

plt.style.use('fivethirtyeight')

#ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

url_stats = 'https://uk.finance.yahoo.com/quote/{}/key-statistics?p={}'
companies = ['TSLA', 'MSFT', 'AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX', 'SPOT', 'ZM', 'DBX', 'KO', 'DIS', 'PEP', 'ADBE', 'SONO', 'AMD', 'NVDA', 'INTC', 'NKE', 'NOK']
stock = 'MSFT'

def datetime_to_float(d):
    return d.timestamp()

def getCSVS():
    #Statistics
    response = requests.get(url_stats.format(stock, stock))
    soup = BeautifulSoup(response.text, 'html.parser')
    pattern = re.compile(r'\s--\sData\s--\s')
    script_data = soup.find('script', text=pattern).contents[0]
    start = script_data.find("context")-2
    json_data = json.loads(script_data[start:-12])

    now = datetime.datetime.today().strftime('%Y-%m-%d')
    # print(now)

    #Historical Stock Data
    stock_url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?'

    params = {
        'start' : now,
        'interval' : '1d',
        'range' : '1y',
        'events' : 'history'
    }

    for s in companies:
        response = requests.get(stock_url.format(s), params=params)
        response.text

        file = StringIO(response.text)
        reader = csv.DictReader(file)
        with open('./csvs/'+s+'.csv', 'w', newline='') as f:
            fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for row in reader:
                # writer.writerow({'Date': datetime_to_float(datetime.strptime(row['Date'], "%Y-%m-%d")), 'Open': row['Open'], 'High': row['High'], 'Low': row['Low'], 'Close': row['Close'], 'Adj Close': row['Adj Close'], 'Volume': row['Volume']})
                writer.writerow({'Date': row['Date'], 'Open': row['Open'], 'High': row['High'], 'Low': row['Low'], 'Close': row['Close'], 'Adj Close': row['Adj Close'], 'Volume': row['Volume']})

def ad_test(dataset):
    dftest = adfuller(dataset, autolag= 'AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num of observations used for adf regression and critical values calculation : ", dftest[3])
    print("5. Critical Values : ")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

def getDataframe(stockIn):
    # Read data

    stock_df = pd.read_csv('csvs/' + stockIn + ".csv", index_col='Date', parse_dates=True)
    print(stock_df.shape)
    print(stock_df.head())

    missing_data = stock_df[stock_df.isna().any(axis=1)]
    print(missing_data)

    print(stock_df)
    stock_df = stock_df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)
    print(stock_df)

    stock_df = stock_df.applymap(np.float)

    return stock_df

def trainModel(stockName):
    stock_df = getDataframe(stockName)
    ad_test(stock_df['Close'])

    #Stepwise to minimse aic value
    # stepwise_fit = auto_arima(stock_df, trace=True, suppress_warnings=True)
    # print(stepwise_fit.summary())

    y_train, y_test = train_test_split(stock_df, test_size=0.2)
    print(y_train.shape)
    print(y_test.shape)

    model=ARIMA(y_train['Close'], order=(1,1,0))
    model=model.fit()
    model.summary()

    #Make predictions on Test Set
    start=len(y_train)
    end=len(y_train)+len(y_test)-1
    pred=model.predict(start=start,end=end, typ='levels')
    pred.index=stock_df.index[start:end+1]
    print(pred)

    #Dataset Mean
    print(y_test['Close'].mean())
    rmse = sqrt(mean_squared_error(pred, y_test['Close']))
    print(rmse)

    # print(pred)


    model2=ARIMA(stock_df['Close'], order=(3,1,3))
    model2=model2.fit()
    print(stock_df.tail())

    return model2

def getPred(numDays, stock_df, stockIn):
    model = trainModel(stockIn)
    n = int(numDays)
    currentDate = datetime.datetime.now()
    predDate = currentDate + datetime.timedelta(days=n)
    print(predDate.strftime('%Y-%m-%d'))

    index_future_dates=pd.date_range(start=currentDate,end=predDate)
    print(index_future_dates)

    pred=model.predict(start=len(stock_df), end=len(stock_df)+n, typ='levels').rename('Prediction')
    pred.index=index_future_dates.strftime('%Y-%m-%d')
    print("------------------------ Pred Info ------------------------")
    print(pred)

    return pred

def getPredGraph(days, stockIn):
    n = int(days)
    stock_df = getDataframe(stockIn)
    pred = getPred(n, stock_df, stockIn)

    plt.figure(figsize=(25,17))
    plt.title(stockIn+' Closing Price Prediction in '+str(days)+' Days')
    plt.plot(pred)
    plt.xlabel('Date', fontsize=18)
    plt.xticks(rotation=90)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig('static/images/predGraph.png')

    return plt

def getProfits(periodIn):
    profits = {}
    period = int(periodIn)
    yest = datetime.datetime.today() - relativedelta(days=1)
    yest = yest.strftime('%Y-%m-%d')
    lastClose = 0

    for c in companies:
        df = web.DataReader(c, data_source='yahoo', start=yest, end=yest)
        lastClose = df['Close'][df.index[0]]
        pred = getPred(period, getDataframe(c), c)
        predClose = pred[period]
        profits[c]=lastClose-predClose

        print("Prediction")
        print(pred[period])
        print("Last Close ")
        print(lastClose)
        print("Predicted Close ")
        print(predClose)

    print(profits)
    return profits


print("--------------------------------TESTING--------------------------------")

# getProfits(30)

