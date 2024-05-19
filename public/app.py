from analysis import *
from extensions import app, db

import os
import random
import threading
import json
from datetime import datetime

from flask import render_template, send_from_directory, url_for, request, redirect, flash, jsonify

version = 'Alpha 15.05.24'

@app.route('/')
def hello_world():
    return render_template('index.html', version=version)

@app.route('/portfolio/')
@app.route('/portfolio/<path:path>', methods=['GET', 'POST'])
def portfolio(path='overview'):
    if path == 'overview':
        stocks = Stock.query.all()
        portfolio_value = 0
        market_value = 0
        for stock in stocks:
            stock.buy_price = round(stock.buy_price, 2)
            portfolio_value += stock.buy_price * stock.shares
            
            stock.current_price = get_stock_data(stock.ticker).iloc[-1]['Close']
            market_value += stock.current_price * stock.shares

        portfolio_value_line_chartData, portfolio_composition_value_pie_chartData, portfolio_composition_shares_pie_chartData = get_stock_chart() 
        
        return render_template('portfolio-overview.html', portfolio_value_line_chartData=portfolio_value_line_chartData, portfolio_composition_value_pie_chartData=portfolio_composition_value_pie_chartData, portfolio_composition_shares_pie_chartData=portfolio_composition_shares_pie_chartData, portfolio_value=round(portfolio_value, 2), market_value=round(market_value, 2), version=version)

    elif path == 'returns':
        stocks = Stock.query.all()
        portfolio_profitloss_line_chartData = returns_graph()

        if portfolio_profitloss_line_chartData != None:
            for stock in stocks:
                stock.current_price = get_stock_data(stock.ticker).iloc[-1]['Close']
                stock.ticker = stock.ticker[:-3]
                stock.buy_price = round(stock.buy_price, 2)
                stock.buy_date = stock.buy_date.strftime('%Y-%m-%d') 

        return render_template('portfolio-returns.html', portfolio_profitloss_line_chartData=portfolio_profitloss_line_chartData, stocks=stocks, version=version)

    elif path == 'management':
        stocks = Stock.query.all()
        if request.method == "POST":
            request_type = request.form['type']
            ticker = request.form['ticker']
            shares = request.form['shares']
            buy_price = request.form['buy_price']
            buy_date = request.form['buy_date']

            if '.JK' not in ticker:
                ticker = ticker + '.JK'
            try:
                info = yf.Ticker(ticker).info
            except:
                flash(('Invalid ticker', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            if not shares.replace('.', '').isdigit() or float(shares) < 0 or float(shares) > 1000000000:
                flash(('Invalid shares', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            if not buy_price.replace('.', '').isdigit() or float(buy_price) < 0 or float(buy_price) > 1000000:
                flash(('Invalid buy price', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            try:
                buy_date = datetime.strptime(buy_date, '%Y-%m-%d').date()
            except ValueError:
                flash(('Invalid buy date format', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            stock = Stock()

            if request_type == 'add':
                if stock.read_stock(ticker):
                    existing_stock = stock.read_stock(ticker)
                    new_shares = existing_stock.shares + float(shares)
                    new_buy_price = ((existing_stock.shares * existing_stock.buy_price) + (float(shares) * float(buy_price))) / new_shares
                    new_buy_date = buy_date
                    stock.update_stock(ticker, shares=new_shares, buy_price=new_buy_price)
                    flash((f'Stock {ticker} has been updated', 'success'))
                    return redirect(url_for('portfolio', path='management'))
                elif not stock.read_stock(ticker):
                    stock.create_stock(ticker, shares, buy_price, buy_date)
                    flash((f'{ticker[:-3]} has been added to your portfolio', 'success'))
                    return redirect(url_for('portfolio', path='management'))
                else:
                    flash(('An error occurred', 'danger'))
                    return redirect(url_for('portfolio', path='management'))

            elif request_type == 'edit':
                if not stock.read_stock(ticker):
                    flash(('Stock not found', 'danger'))
                    return redirect(url_for('portfolio', path='management'))
                else:
                    stock.update_stock(ticker, shares, buy_price, buy_date)
                    flash((f'Stock {ticker[:-3]} has been updated', 'success'))
                    return redirect(url_for('portfolio', path='management'))
            
            elif request_type == 'delete':
                if not stock.read_stock(ticker):
                    flash(('Stock not found', 'danger'))
                    return redirect(url_for('portfolio', path='management'))
                else:
                    stock.delete_stock(ticker)
                    flash((f'Stock {ticker[:-3]} has been deleted', 'success'))
                    return redirect(url_for('portfolio', path='management'))

        for stock in stocks:
            stock.ticker = stock.ticker[:-3]
            stock.buy_price = round(stock.buy_price, 2)
            stock.buy_date = stock.buy_date.strftime('%Y-%m-%d')

        return render_template('portfolio-management.html', stocks=stocks, version=version)
    else:
        return "Invalid path"

@app.route('/analysis/')
@app.route('/analysis/<path:path>', methods=['GET', 'POST'])
def analysis(path='overview'):
    if path == 'overview':
        return render_template('analysis-overview.html', version=version)

    elif path == 'ownership':
        if request.method == 'POST':
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']

            if request_type == "ownership":
                stock_ownership_graphData = ownership_graph(ticker)
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_ownership_graphData

        return render_template('analysis-ownership.html', version=version)

    elif path == 'performance':
        if request.method == 'POST':
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            riskFreeRate = data['riskFreeRate']


        return render_template('analysis-performance.html', version=version)

    elif path == 'distribution':
        if request.method == 'POST':
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            
            if '.JK' not in ticker:
                ticker = ticker + '.JK'
            try:
                info = yf.Ticker(ticker).info
            except:
                return jsonify({'error': 'Invalid ticker'})

            if request_type == "distribution":
                stockDistributionGraphData = distribution_graph(ticker)
            else:
                return jsonify({'error': 'Invalid request type'})

            return stockDistributionGraphData

        return render_template('analysis-distribution.html', version=version)
    
    elif path == 'optimization':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            numPortfolios = data['numPortfolios']
            riskFreeRate = data['riskFreeRate']

            cleaned_ticker = []
            for stock in ticker:    
                if stock == "portfolio":
                    cleaned_ticker = stock
                    break
                elif '.JK' not in stock:
                    stock = stock + '.JK'
                    try:
                        info = yf.Ticker(stock).info
                    except:
                        jsonify({'error': 'Invalid ticker'})

                cleaned_ticker.append(stock)

            if len(cleaned_ticker) == 0:
                return jsonify({'error': 'Select at least 2 stocks'})
            elif riskFreeRate == '' or numPortfolios not in range(1, 10001):
                return jsonify({'error': 'Number of portfolios must be between 1 and 10000'})
            elif riskFreeRate == '' or float(riskFreeRate) < 0 or float(riskFreeRate) > 1:
                return jsonify({'error': 'Risk-free rate must be between 0 and 1'})

            if request_type == 'optimization':
                stock_optimization_graphData, maxSharpeWeights, minVolWeights = optimization_graph(cleaned_ticker, numPortfolios, riskFreeRate)
            else:
                return jsonify({'error': 'Invalid request type'})

            if stock_optimization_graphData == "1":
                return jsonify({'error': 'Please add more than one stock to be analyzed'})
            elif stock_optimization_graphData == "2":
                return jsonify({'error': 'At least one of the assets must have an expected return exceeding the risk-free rate'})
            elif stock_optimization_graphData == "3":
                return jsonify({'error': 'An unknown error occurred'})
            else:
                maxSharpeWeights_json = json.dumps(maxSharpeWeights)
                minVolWeights = json.dumps(minVolWeights)
                stock_optimization_graphData = stock_optimization_graphData[:-1] + f', "maxSharpeWeights": {maxSharpeWeights_json} ' + f', "minVolWeights": {minVolWeights} ' + '}'

                return stock_optimization_graphData

        return render_template('analysis-optimization.html', stock_optimization_graphData=None, version=version)

    elif path == 'fundamental':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            cleaned_ticker = []

            for stock in ticker:
                if stock == "portfolio":
                    cleaned_ticker = stock
                    break
                elif '.JK' not in stock:
                    stock = stock + '.JK'
                    try:
                        info = yf.Ticker(stock).info
                    except:
                        jsonify({'error': 'Invalid ticker'})

                cleaned_ticker.append(stock)

            if len(cleaned_ticker) == 0:
                return jsonify({'error': 'Select at least 1 stock'})

            if request_type == 'fundamental':
                stock_fundamental_data = get_stock_fundamental(cleaned_ticker)
            else:
                return jsonify({'error': 'Invalid request type'})

            if stock_fundamental_data == "1":
                return jsonify({'error': 'Select at least 1 stock'})

            return jsonify(stock_fundamental_data)

        return render_template('analysis-fundamental.html', version=version)

    elif path == 'decomposition':
        if request.method == "POST":
            data = request.get_json()
            ticker = data['ticker']
            period = data['period'] 

            period = int(period)

            if ticker == '':
                return jsonify({'error': 'Please enter a ticker'})
            elif '.JK' not in ticker:
                ticker = ticker + '.JK'

            try:
                info = yf.Ticker(ticker).info
            except:
                return jsonify({'error': 'Invalid ticker'})

            stock_decomposition_graphData = decomposition_graph(ticker, period)

            if stock_decomposition_graphData == "1":
                return jsonify({'error': 'Please enter a valid ticker'})
            elif stock_decomposition_graphData == "2":
                return jsonify({'error': 'Please enter a valid period'})

            return stock_decomposition_graphData

        return render_template('analysis-decomposition.html', version=version)
        
    elif path == 'garch':
        if request.method == "POST":
            data = request.get_json()
            ticker = data['ticker']
            request_type = data['type']

            if '.JK' not in ticker:
                ticker = ticker + '.JK'
            try:
                info = yf.Ticker(ticker).info
            except:
                return jsonify({'error': 'Invalid ticker'})

            if request_type == 'garch':
                stock_garch_graphData = garch_graph(ticker)
                if stock_garch_graphData == "1":
                    return jsonify({'error': 'Please enter a valid ticker'})
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_garch_graphData

        return render_template('analysis-garch.html', version=version)

    else:
        return "Invalid path"
        
@app.route('/assets/<path:filename>')
def send_assets(filename):
    return send_from_directory('assets', filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)