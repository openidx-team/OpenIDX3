"""
This module contains the main Flask application for the OpenIDX3 project.
"""

import json
from datetime import datetime

from extensions import app, db
from validator import validate_stock
from analysis import (
    Stock,
    get_stock_data,
    get_stock_fundamental,
    get_stock_chart,
    returns_graph,
    decomposition_graph,
    garch_graph,
    optimization_graph,
    distribution_graph,
    ownership_graph,
    performance_graph,
    get_stock_sentiment,
    get_stock_prediction,
    get_portfolio_analysis
)

from flask import render_template, send_from_directory, url_for, request, redirect, flash, jsonify

VERSION = 'Alpha 29.05.24'

@app.route('/')
def index():
    """
    This function renders the index page of the website.
    """
    return render_template('index.html', version=VERSION)

@app.route('/portfolio/')
@app.route('/portfolio/<path:path>', methods=['GET', 'POST'])
def portfolio(path='overview'):
    """
    This function renders the portfolio page of the website.
    """
    if path == 'overview':
        stocks = Stock.query.all()
        portfolio_value = 0
        market_value = 0

        for stock in stocks:
            stock.buy_price = round(stock.buy_price, 2)
            portfolio_value += stock.buy_price * stock.shares

            stock.current_price = get_stock_data(stock.ticker).iloc[-1]['Close']
            market_value += stock.current_price * stock.shares

        portfolio_value_line_chart_data, portfolio_composition_value_pie_chart_data, portfolio_composition_shares_pie_chart_data = get_stock_chart()

        return render_template(
            'portfolio-overview.html',
            portfolio_value_line_chartData=portfolio_value_line_chart_data,
            portfolio_composition_value_pie_chartData=portfolio_composition_value_pie_chart_data,
            portfolio_composition_shares_pie_chartData=portfolio_composition_shares_pie_chart_data,
            portfolio_value=round(portfolio_value, 2),
            market_value=round(market_value, 2),
            version=VERSION
        )

    if path == 'returns':
        stocks = Stock.query.all()
        portfolio_profitloss_line_chart_data = returns_graph()

        if portfolio_profitloss_line_chart_data is not None:
            for stock in stocks:
                stock.current_price = get_stock_data(stock.ticker).iloc[-1]['Close']
                stock.ticker = stock.ticker[:-3]
                stock.buy_price = round(stock.buy_price, 2)
                stock.buy_date = stock.buy_date.strftime('%Y-%m-%d')

        return render_template(
            'portfolio-returns.html',
            portfolio_profitloss_line_chartData=portfolio_profitloss_line_chart_data,
            stocks=stocks,
            version=VERSION
        )

    if path == 'management':
        stocks = Stock.query.all()
        if request.method == "POST":
            request_type = request.form['type']
            ticker = request.form['ticker']
            shares = request.form['shares']
            buy_price = request.form['buy_price']
            buy_date = request.form['buy_date']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                flash((f'{ticker[:-3]} is an invalid ticker', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            if (not shares.replace('.', '').isdigit() or
                    float(shares) < 0 or
                    float(shares) > 1000000000):
                flash(('Invalid shares', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            if (not buy_price.replace('.', '').isdigit() or
                    float(buy_price) < 0 or
                    float(buy_price) > 1000000):
                flash(('Invalid buy price', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            try:
                buy_date = datetime.strptime(buy_date, '%Y-%m-%d').date()
            except ValueError:
                flash(('Invalid buy date format', 'danger'))
                return redirect(url_for('portfolio', path='management'))
            
            if buy_date > datetime.now().date():
                flash(('Buy date cannot be in the future', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            stock = Stock()

            if request_type == 'add':
                if stock.read_stock(ticker):
                    existing_stock = stock.read_stock(ticker)
                    new_shares = existing_stock.shares + float(shares)
                    new_buy_price = ((existing_stock.shares * existing_stock.buy_price) +
                        (float(shares) * float(buy_price))) / new_shares
                    new_buy_date = buy_date
                    stock.update_stock(ticker, shares=new_shares, buy_price=new_buy_price, buy_date=new_buy_date)
                    flash((f'Stock {ticker[:-3]} has been updated', 'success'))
                    return redirect(url_for('portfolio', path='management'))

                if not stock.read_stock(ticker):
                    stock.create_stock(ticker, shares, buy_price, buy_date)
                    flash((f'{ticker[:-3]} has been added to your portfolio', 'success'))
                    return redirect(url_for('portfolio', path='management'))

                flash(('An error occurred', 'danger'))
                return redirect(url_for('portfolio', path='management'))

            if request_type == 'edit':
                if not stock.read_stock(ticker):
                    flash(('Stock not found', 'danger'))
                    return redirect(url_for('portfolio', path='management'))

                stock.update_stock(ticker, shares, buy_price, buy_date)
                flash((f'Stock {ticker[:-3]} has been updated', 'success'))
                return redirect(url_for('portfolio', path='management'))

            if request_type == 'delete':
                if not stock.read_stock(ticker):
                    flash(('Stock not found', 'danger'))
                    return redirect(url_for('portfolio', path='management'))

                stock.delete_stock(ticker)
                flash((f'Stock {ticker[:-3]} has been deleted', 'success'))
                return redirect(url_for('portfolio', path='management'))

        for stock in stocks:
            stock.ticker = stock.ticker[:-3]
            stock.buy_price = round(stock.buy_price, 2)
            stock.buy_date = stock.buy_date.strftime('%Y-%m-%d')

        return render_template('portfolio-management.html', stocks=stocks, version=VERSION)

    return "Invalid path"

@app.route('/analysis/')
@app.route('/analysis/<path:path>', methods=['GET', 'POST'])
def analysis(path='overview'):
    """
    This function renders the analysis page of the website.
    """
    if path == 'overview':
        portfolio_analysis_overview = get_portfolio_analysis(period=365, riskFreeRate=0.065)

        if portfolio_analysis_overview == "1":
            return render_template(
                'analysis-overview.html',
                portfolio_analysis_overview=None,
                overall_assessment=None,
                version=VERSION
            )

        good = 0
        bad = 0
        neutral = 0

        for metric_info in portfolio_analysis_overview.values():
            condition = metric_info['condition']
            if condition == 'Good':
                good += 1
            elif condition == 'Bad':
                bad += 1
            elif condition == 'Neutral':
                neutral += 1

        if good > bad and good > neutral:
            overall_assessment = 'Good'
        elif bad > good and bad > neutral:
            overall_assessment = 'Bad'
        else:
            overall_assessment = 'Neutral'

        return render_template(
            'analysis-overview.html',
            portfolio_analysis_overview=portfolio_analysis_overview,
            overall_assessment = overall_assessment,
            version=VERSION
        )

    if path == 'ownership':
        if request.method == 'POST':
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if request_type == "ownership":
                stock_ownership_graph_data = ownership_graph(ticker)
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_ownership_graph_data

        return render_template('analysis-ownership.html', version=VERSION)

    if path == 'performance':
        if request.method == 'POST':
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            risk_free_rate = data['riskFreeRate']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if len(ticker) == 0:
                return jsonify({'error': 'Select at least 1 stock'})

            if risk_free_rate == '' or float(risk_free_rate) < 0 or float(risk_free_rate) > 1:
                return jsonify({'error': 'Risk-free rate must be between 0 and 1'})

            if request_type == "performance":
                stock_performance_graph_data = performance_graph(ticker, risk_free_rate)
                if stock_performance_graph_data == "1":
                    return jsonify({'error': 'Select at least 1 stock'})
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_performance_graph_data

        return render_template('analysis-performance.html', version=VERSION)

    if path == 'distribution':
        if request.method == 'POST':
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if request_type == "distribution":
                stock_distribution_graph_data = distribution_graph(ticker)
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_distribution_graph_data

        return render_template('analysis-distribution.html', version=VERSION)

    if path == 'optimization':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            num_portfolios = data['numPortfolios']
            risk_free_rate = data['riskFreeRate']

            cleaned_ticker = []
            for stock in ticker:
                if stock == "portfolio":
                    cleaned_ticker = "portfolio"
                    break

                is_valid, stock = validate_stock(stock)
                if not is_valid:
                    return jsonify({'error': f'{stock[:-3]} is an invalid ticker'})

                cleaned_ticker.append(stock)

            if len(cleaned_ticker) == 0:
                return jsonify({'error': 'Select at least 2 stocks'})

            if risk_free_rate == '' or num_portfolios not in range(1, 10001):
                return jsonify({'error': 'Number of portfolios must be between 1 and 10000'})

            if risk_free_rate == '' or float(risk_free_rate) < 0 or float(risk_free_rate) > 1:
                return jsonify({'error': 'Risk-free rate must be between 0 and 1'})

            if request_type == 'optimization':
                stock_optimization_graphData, maxSharpeWeights, minVolWeights = optimization_graph(cleaned_ticker, num_portfolios, risk_free_rate)
            else:
                return jsonify({'error': 'Invalid request type'})

            if stock_optimization_graphData == "1":
                return jsonify({'error': 'The stock in your portfolio must be more than 1'})
            if stock_optimization_graphData == "2":
                return jsonify({
                    'error': 'At least one of the assets must have an expected return exceeding the risk-free rate'
                })
            if stock_optimization_graphData == "3":
                return jsonify({'error': 'An unknown error occurred'})

            maxSharpeWeights_json = json.dumps(maxSharpeWeights)
            minVolWeights = json.dumps(minVolWeights)
            stock_optimization_graphData = (
                stock_optimization_graphData[:-1] +
                f', "maxSharpeWeights": {maxSharpeWeights_json} ' +
                f', "minVolWeights": {minVolWeights} ' +
                '}'
            )

            return stock_optimization_graphData

        return render_template(
            'analysis-optimization.html', 
            stock_optimization_graphData=None,
            version=VERSION
        )

    if path == 'fundamental':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']

            cleaned_ticker = []
            for stock in ticker:
                if stock == "portfolio":
                    cleaned_ticker = "portfolio"
                    break
                
                is_valid, stock = validate_stock(stock)
                if not is_valid:
                    return jsonify({'error': f'{stock[:-3]} is an invalid ticker'})

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

        return render_template('analysis-fundamental.html', version=VERSION)

    if path == 'decomposition':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            period = data['period']

            period = int(period)

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if request_type == 'decomposition':
                stock_decomposition_graph_data = decomposition_graph(ticker, period)
            else:
                return jsonify({'error': 'Invalid request type'})

            if stock_decomposition_graph_data == "1":
                return jsonify({'error': 'Please enter a valid ticker'})
            if stock_decomposition_graph_data == "2":
                return jsonify({'error': 'Please enter a valid period'})

            return stock_decomposition_graph_data

        return render_template('analysis-decomposition.html', version=VERSION)

    if path == 'garch':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if request_type == 'garch':
                stock_garch_graph_data = garch_graph(ticker)
                if stock_garch_graph_data == "1":
                    return jsonify({'error': 'Please enter a valid ticker'})
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_garch_graph_data

        return render_template('analysis-garch.html', version=VERSION)

    if path == 'sentiment':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if request_type == 'sentiment':
                news, score = get_stock_sentiment(ticker)
                return jsonify({'news': news, 'score': score})
            else:
                return jsonify({'error': 'Invalid request type'})

        return render_template('analysis-sentiment.html', version=VERSION)

    if path == 'prediction':
        if request.method == "POST":
            data = request.get_json()
            request_type = data['type']
            ticker = data['ticker']
            model = data['model']

            is_valid, ticker = validate_stock(ticker)
            if not is_valid:
                return jsonify({'error': f'{ticker[:-3]} is an invalid ticker'})

            if request_type == 'prediction':
                stock_prediction_graph_data = get_stock_prediction(ticker, model)
                if stock_prediction_graph_data == "1":
                    return jsonify({'error': 'Please enter a valid model'})
                if stock_prediction_graph_data == "2":
                    return jsonify({'error': f'{ticker[:-3]} does not have enough data to make a prediction using LSTM'})
            else:
                return jsonify({'error': 'Invalid request type'})

            return stock_prediction_graph_data

        return render_template('analysis-prediction.html', version=VERSION)

    return "Invalid path"

@app.route('/assets/<path:filename>')
def send_assets(filename):
    """
    This function sends the assets to the client.
    """
    return send_from_directory('assets', filename)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    app.run(debug=True)
