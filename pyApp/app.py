import sys
from functions import createHistoricalGraph, getCSVS
from flask import Flask, render_template, url_for, request, redirect
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas_datareader as web

companies = ['AAPL', 'ADBE', 'AMD', 'AMZN', 'DBX', 'DIS', 'FB', 'GOOG', 'INTC', 'KO', 'MSFT', 'NFLX', 'NKE', 'NOK', 'NVDA', 'PEP', 'SONO', 'SPOT', 'TSLA', 'ZM']

app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"

@app.route('/stock', methods=['POST', 'GET'])
@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        stock = request.form['stock']
        per = request.form['per']
        timeperiod = request.args.get('timeperiod')
        return redirect(url_for('stock_calc', stock=stock, per=timeperiod))
    else:
        getCSVS()
        return render_template('stock.html')

@app.route("/history", methods=['POST', 'GET'])

def history():
    if request.method == 'POST':
        stock = request.form['stock']
        # per = request.form['per']
        print(stock, file=sys.stderr)
        createHistoricalGraph(stock)
        return render_template('history.html')
    else:
        return render_template('history.html')

@app.route("/profit", methods=['POST', 'GET'])
def profit():
    return render_template('profit.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
    app.run(debug=True)