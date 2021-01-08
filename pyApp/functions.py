import pandas_datareader as web
import matplotlib.pyplot as plt
from arimamodel import getDataframe, getPredGraph, trainModel, getPred, getCSVS
from datetime import datetime
import re
import json
import csv
from io import StringIO
from bs4 import BeautifulSoup
import requests
from dateutil.relativedelta import relativedelta
plt.style.use('fivethirtyeight')

url_stats = 'https://uk.finance.yahoo.com/quote/{}/key-statistics?p={}'
companies = ['TSLA', 'MSFT', 'AAPL', 'AMZN', 'FB', 'GOOG', 'NFLX', 'SPOT', 'ZM', 'DBX', 'KO', 'DIS', 'PEP', 'ADBE', 'SONO', 'AMD', 'NVDA', 'INTC', 'NKE', 'NOK']
stock = 'TSLA'

def getCSVS():
    #Statistics
    response = requests.get(url_stats.format(stock, stock))
    soup = BeautifulSoup(response.text, 'html.parser')
    pattern = re.compile(r'\s--\sData\s--\s')
    script_data = soup.find('script', text=pattern).contents[0]
    start = script_data.find("context")-2
    json_data = json.loads(script_data[start:-12])

    now = datetime.datetime.today().strftime('%Y-%m-%d')

    #Historical Stock Data
    stock_url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?'

    params = {
        'start' : now,
        'interval' : '1d',
        'range' : '5y',
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

def createHistoricalGraph(stockIn):
    now = datetime.today().strftime('%Y-%m-%d')
    oneYrAgo = datetime.now() - relativedelta(years=1)
    stock = stockIn

    # Get the stock quote
    df = web.DataReader(stock, data_source='yahoo', start=oneYrAgo, end=now)

    plt.figure(figsize=(16, 8))
    plt.title('Closing Price History for '+stock)
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.savefig('static/images/plot.png')

def predictGraph(stockIn, perIn):
    getPredGraph(perIn, stockIn)
