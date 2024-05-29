from extensions import db
from flask import jsonify

import os
import random
import threading
import json
from datetime import datetime, timedelta
import locale
import re

import yfinance as yf

import pandas as pd
import numpy as np

from arch import arch_model
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa as tsa
from statsmodels.tsa.stattools import acf
from statsmodels.tools.eval_measures import aic
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from pmdarima.model_selection import train_test_split
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from transformers import AutoTokenizer, TFRobertaForSequenceClassification
from transformers import pipeline

from deep_translator import GoogleTranslator

import requests
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
model = TFRobertaForSequenceClassification.from_pretrained("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    shares = db.Column(db.Float, nullable=False)
    buy_price = db.Column(db.Float, nullable=False)
    buy_date = db.Column(db.Date, nullable=False)
    sell_price = db.Column(db.Float, nullable=True)
    sell_date = db.Column(db.Date, nullable=True)

    def __repr__(self):
        return f'<Stock {self.ticker}>'

    def create_stock(self, ticker, shares, buy_price, buy_date, sell_price=None, sell_date=None):
        new_stock = Stock(ticker=ticker, shares=shares, buy_price=buy_price, buy_date=buy_date, sell_price=sell_price, sell_date=sell_date)
        db.session.add(new_stock)
        db.session.commit()

    def read_stock(self, ticker):
        stock = Stock.query.filter_by(ticker=ticker).first()
        return stock

    def update_stock(self, ticker, shares=None, buy_price=None, buy_date=None, sell_price=None, sell_date=None):
        stock = Stock.query.filter_by(ticker=ticker).first()
        if shares:
            stock.shares = shares
        if buy_price:
            stock.buy_price = buy_price
        if buy_date:
            stock.buy_date = buy_date
        if sell_price:
            stock.sell_price = sell_price
        if sell_date:
            stock.sell_date = sell_date
        db.session.commit()

    def delete_stock(self, ticker):
        stock = Stock.query.filter_by(ticker=ticker).first()
        db.session.delete(stock)
        db.session.commit()

def get_stock_data(tickers):
    if isinstance(tickers, list):
        data_frames = pd.DataFrame()
        for ticker in tickers:
            ticker = ticker.strip()
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_file = os.path.join(cache_dir, f'{ticker}.csv')
            if os.path.exists(cache_file):
                data = pd.read_csv(cache_file, index_col=0)
            else:
                try:
                    data = yf.download(ticker)
                except Exception as e:
                    return "1"
                data.to_csv(cache_file)
            
            data_frames[ticker] = data['Close']

        return data_frames
    else:
        ticker = tickers.strip()
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, f'{ticker}.csv')
        if os.path.exists(cache_file):
            data = pd.read_csv(cache_file, index_col=0)
        else:
            data = yf.download(ticker)
            data.to_csv(cache_file)
        return data

def get_stock_fundamental(stocks):
    stock_tickers = []
    if stocks == "portfolio":
        stocks = Stock.query.all()
        stock_tickers = [stock.ticker for stock in stocks]
    else:
        stock_tickers = stocks

    if stock_tickers == []:
        return "1"

    fundamental_data = {}
    for stock in stock_tickers:
        info = yf.Ticker(stock).info
        fundamental_data[stock] = {
            'ticker': stock[:-3].upper(),
            'priceToBook': round(info.get('priceToBook', None), 2) if info.get('priceToBook', None) is not None else None,
            'trailingPE': round(info.get('trailingPE', None), 2) if info.get('trailingPE', None) is not None else None,
            'dividendYield': str(round(info.get('dividendYield', None) * 100, 2)) + '%' if info.get('dividendYield', None) is not None else None,
            'profitMargins': str(round(info.get('profitMargins', None) * 100, 2)) + '%' if info.get('profitMargins', None) is not None else None,
            'operatingMargins': str(round(info.get('operatingMargins', None) * 100, 2)) + '%' if info.get('operatingMargins', None) is not None else None,
            'returnOnAssets': str(round(info.get('returnOnAssets', None) * 100, 2)) + '%' if info.get('returnOnAssets', None) is not None else None,
            'returnOnEquity': str(round(info.get('returnOnEquity', None) * 100, 2)) + '%' if info.get('returnOnEquity', None) is not None else None
        }

    return fundamental_data

def get_stock_chart():
    stocks = Stock.query.all()

    if stocks == []:
        return None, None, None

    portfolio_value = pd.DataFrame()
    portfolio_capital = pd.DataFrame()
    portfolio_pnl = pd.DataFrame()

    for stock in stocks:
        buy_date = pd.to_datetime(stock.buy_date)
        data = get_stock_data(stock.ticker)
        data.index = pd.to_datetime(data.index)
        data['Value'] = data['Close'] * stock.shares
        data = data[data.index >= buy_date]
        data = data[['Value']]
        
        portfolio_value = pd.merge(portfolio_value, data, how='outer', left_index=True, right_index=True)
        portfolio_value = portfolio_value.fillna(0)
        portfolio_value = portfolio_value.sum(axis=1)
        portfolio_value = pd.DataFrame(portfolio_value, columns=['Value'])

        capital = pd.DataFrame(data={'Date': [buy_date], 'Capital': [stock.buy_price * stock.shares]})
        portfolio_capital = pd.concat([portfolio_capital, capital])

    portfolio_capital = portfolio_capital.set_index('Date')
    portfolio_capital = portfolio_capital.sort_index()
    portfolio_capital = portfolio_capital.cumsum()
    portfolio_capital = portfolio_capital.loc[~portfolio_capital.index.duplicated(keep='last')]
    portfolio_capital = portfolio_capital.reindex(portfolio_value.index, method='ffill')

    for i in range(len(portfolio_value.index)):
        portfolio_pnl.loc[portfolio_value.index[i], 'PNL'] = portfolio_value['Value'].iloc[i] - portfolio_capital['Capital'].iloc[i]
    
    fig = px.line(portfolio_value, x=portfolio_value.index, y='Value', title='Portfolio Value Over Time', template="seaborn", color_discrete_sequence=['yellow'])
    fig.add_scatter(x=portfolio_value.index, y=portfolio_value['Value'].where(portfolio_pnl['PNL'] < 0), mode='lines', name='Negative Value', line_color='red')
    fig.add_scatter(x=portfolio_value.index, y=portfolio_value['Value'].where(portfolio_pnl['PNL'] > 0), mode='lines', name='Positive Value', line_color='green')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Value')
    fig.update_layout(showlegend=False)
    portfolio_value_line_chartData = fig.to_json()

    latest_data = []
    latest_data_shares = []
    for stock in stocks:
        data = get_stock_data(stock.ticker)
        data.index = pd.to_datetime(data.index)
        latest_date = data.index.max()
        latest_close_price = data.loc[latest_date, 'Close']
        stock_value = latest_close_price * stock.shares
        latest_data.append({'ticker': stock.ticker, 'Value': stock_value})
        latest_data_shares.append({'ticker': stock.ticker, 'Shares': stock.shares})

    latest_data = pd.DataFrame(latest_data)

    stock_color = {stock: px.colors.qualitative.Plotly[i] for i, stock in enumerate(stocks)}

    fig2 = px.pie(latest_data, values='Value', names='ticker', title='Portfolio Composition By Value', template="seaborn", color=stock_color)
    portfolio_composition_value_pie_chartData = fig2.to_json()

    latest_data_shares = pd.DataFrame(latest_data_shares)

    fig3 = px.pie(latest_data_shares, values='Shares', names='ticker', title='Portfolio Composition By Shares', template="seaborn", color=stock_color)
    portfolio_composition_shares_pie_chartData = fig3.to_json()

    return portfolio_value_line_chartData, portfolio_composition_value_pie_chartData, portfolio_composition_shares_pie_chartData

def returns_graph():
    stocks = Stock.query.all()
    if stocks == []:
        return None
        
    profit_loss = pd.DataFrame()
    for stock in stocks:
        buy_date = pd.to_datetime(stock.buy_date)
        data = get_stock_data(stock.ticker)
        data.index = pd.to_datetime(data.index)
        data['PNL'] = (data['Close'] - stock.buy_price) * stock.shares 
        data = data[data.index >= buy_date]
        data = data[['PNL']]

        profit_loss = pd.merge(profit_loss, data, how='outer', left_index=True, right_index=True)
        profit_loss = profit_loss.fillna(0)
        profit_loss = profit_loss.sum(axis=1)
        profit_loss = pd.DataFrame(profit_loss, columns=['PNL'])

    fig = px.line(profit_loss, x=profit_loss.index, y='PNL', title='Portfolio Profit/Loss Over Time', template="seaborn", color_discrete_sequence=['yellow'])
    fig.add_scatter(x=profit_loss.index, y=profit_loss['PNL'].where(profit_loss['PNL'] < 0), mode='lines', name='Negative P&L', line_color='red')
    fig.add_scatter(x=profit_loss.index, y=profit_loss['PNL'].where(profit_loss['PNL'] > 0), mode='lines', name='Positive P&L', line_color='green')
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Profit/Loss')
    fig.update_layout(showlegend=False)
    portfolio_profitloss_line_chartData = fig.to_json()

    return portfolio_profitloss_line_chartData

def get_portfolio_analysis(period, riskFreeRate):
    stocks = Stock.query.all()
    if stocks == []:
        return "1"
    
    stock_values = []
    for stock in stocks:
        data = get_stock_data(stock.ticker)
        data.index = pd.to_datetime(data.index)
        latest_date = data.index.max()
        latest_close_price = data.loc[latest_date, 'Close']
        stock_value = latest_close_price * stock.shares
        stock_values.append(stock_value)

    tickers = [stock.ticker for stock in stocks]
    stock_data = yf.download(tickers, start="2020-01-01", end="2023-01-01")

    stock_close = stock_data['Close']
    stock_returns = np.log(stock_close.pct_change() + 1)
    market_data = yf.download('^JKSE', start=stock_close.index.min(), end=stock_close.index.max())
    market_close = market_data['Close']
    market_returns = np.log(market_close.pct_change() + 1)

    if len(stocks) > 1:
        portfolio_value = sum(stock_values)
        portfolio_weights = [stock_value / portfolio_value for stock_value in stock_values]

        portfolio_returns = stock_returns.dot(portfolio_weights)

        riskFreeRate_daily = riskFreeRate / period
        stock_returns_excess = stock_returns.subtract(riskFreeRate_daily, axis=0)
        portfolio_returns_excess = portfolio_returns - riskFreeRate_daily
        market_returns_excess = market_returns - riskFreeRate_daily

        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(period)

        cum_ret = (1 + portfolio_returns).cumprod()
        cum_roll_max = cum_ret.cummax()
        drawdown = cum_ret / cum_roll_max - 1
        max_drawdown = drawdown.min()

        mean_portfolio = portfolio_returns.mean()
        std_dev_portfolio = portfolio_returns.std()
        portfolio_VaR = mean_portfolio - std_dev_portfolio * stats.norm.ppf(0.05)

        portfolio_CVaR = portfolio_returns[portfolio_returns <= portfolio_VaR].mean()

        cov_matrix = stock_returns_excess.cov()
        market_var = market_returns_excess.var()
        beta = np.dot(portfolio_weights, cov_matrix.dot(portfolio_weights)) / market_var

        alpha = mean_portfolio - riskFreeRate_daily - beta * (market_returns_excess.mean() - riskFreeRate_daily)

        jensen_alpha = portfolio_returns.mean() - riskFreeRate_daily - beta * (market_returns.mean() - riskFreeRate_daily)

        sharpe_ratio = (portfolio_returns.mean() - riskFreeRate_daily) / portfolio_volatility

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std_dev = downside_returns.std()
        sortino_ratio = (portfolio_returns.mean() - riskFreeRate_daily) / downside_std_dev * np.sqrt(period)

        treynor_ratio = (portfolio_returns.mean() - riskFreeRate_daily) / beta
    else:
        portfolio_value = stock_values[0]
        portfolio_returns = stock_returns

        riskFreeRate_daily = riskFreeRate / period
        stock_returns_excess = stock_returns.subtract(riskFreeRate_daily, axis=0)
        portfolio_returns_excess = portfolio_returns - riskFreeRate_daily
        market_returns_excess = market_returns - riskFreeRate_daily

        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(period)

        cum_ret = (1 + portfolio_returns).cumprod()
        cum_roll_max = cum_ret.cummax()
        drawdown = cum_ret / cum_roll_max - 1
        max_drawdown = drawdown.min()

        mean_portfolio = portfolio_returns.mean()
        std_dev_portfolio = portfolio_returns.std()
        portfolio_VaR = mean_portfolio - std_dev_portfolio * stats.norm.ppf(0.05)

        portfolio_CVaR = portfolio_returns[portfolio_returns <= portfolio_VaR].mean()

        cov_matrix = stock_returns_excess.cov(stock_returns_excess) 
        market_var = market_returns_excess.var()
        beta = cov_matrix / market_var

        alpha = mean_portfolio - riskFreeRate_daily - beta * (market_returns_excess.mean() - riskFreeRate_daily)

        jensen_alpha = portfolio_returns.mean() - riskFreeRate_daily - beta * (market_returns.mean() - riskFreeRate_daily)

        sharpe_ratio = (portfolio_returns.mean() - riskFreeRate_daily) / portfolio_volatility

        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std_dev = downside_returns.std()
        sortino_ratio = (portfolio_returns.mean() - riskFreeRate_daily) / downside_std_dev * np.sqrt(period)

        treynor_ratio = (portfolio_returns.mean() - riskFreeRate_daily) / beta
    
    def get_condition(value):
        rounded_value = round(value, 2)
        if rounded_value > 0:
            return "Good"
        elif rounded_value < 0:
            return "Bad"
        else:
            return "Neutral"

    analysis = {
        'Portfolio Volatility': portfolio_volatility,
        'Portfolio Max Drawdown': max_drawdown,
        'Portfolio Value at Risk': portfolio_VaR,
        'Portfolio Conditional Value at Risk': portfolio_CVaR,
        'Portfolio Beta': beta,
        'Portfolio Alpha': alpha,
        'Portfolio Jensen\'s Alpha': jensen_alpha,
        'Portfolio Sharpe Ratio': sharpe_ratio,
        'Portfolio Sortino Ratio': sortino_ratio,
        'Portfolio Treynor Ratio': treynor_ratio
    }
    
    analysis_with_conditions = {metric: {'value': round(value, 2), 'condition': get_condition(value)} for metric, value in analysis.items()}
    
    return analysis_with_conditions

def decomposition_graph(stock, period):
    data = get_stock_data(stock)
    returns = np.log(data[['Close']].dropna())
    prices = data[['Close']]

    periods_to_test = [30, 60, 90, 180, 365, 730]

    if period not in periods_to_test:
        return "2"

    result = tsa.seasonal.STL(returns.squeeze(), period=period).fit()
    trend = result.trend
    seasonal = result.seasonal
    mean_seasonal = seasonal.mean()
    resid = result.resid
    mean_resid = resid.mean()

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "Trend", "Seasonal", "Residual"))
    fig.update_layout(title_text=f"{stock[:-3].upper()} Stock Decomposition Analysis, {period} Days Period", showlegend=False)

    fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Stock Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=trend.index, y=trend.round(2), mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal.round(2), mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=[mean_seasonal] * len(seasonal), mode='lines', name='Mean Seasonal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=resid.index, y=resid.round(2), mode='lines', name='Residual'), row=4, col=1)
    fig.add_trace(go.Scatter(x=resid.index, y=[mean_resid] * len(resid), mode='lines', name='Mean Residual', line=dict(color='red')), row=4, col=1)

    return fig.to_json()

def garch_graph(stock):
    data = get_stock_data(stock)

    if isinstance(data, pd.DataFrame):
        pass
    else:
        return "1"

    returns = data[['Close']].pct_change().dropna()
    returns = returns * 100
    prices = data[['Close']]
    
    p_ = range(5)
    q_ = range(5)
    aic_values = []
    for p in p_:
        for q in q_:
            try:
                model = arch_model(returns, vol='Garch', p=p, q=q)
                model_fit = model.fit(disp='off')
                aic_values.append((p, q, model_fit.aic))
            except Exception as e:
                continue

    p, q, _ = min(aic_values, key=lambda x:x[2])

    model = arch_model(returns, vol='Garch', p=p, q=q)
    results = model.fit(disp='off')
    results_mean = results.conditional_volatility.mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Stock Price", "Conditional Volatility"))
    fig.update_layout(title_text=f"{stock[:-3].upper()} Stock Volatility Analysis, p={p} and q={q}", showlegend=False)

    fig.add_trace(go.Scatter(x=prices.index, y=prices['Close'], mode='lines', name='Stock Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=prices.index, y=results.conditional_volatility, mode='lines', name='Conditional Volatility'), row=2, col=1)
    mean_volatility = [results_mean]*len(prices.index)
    fig.add_trace(go.Scatter(x=prices.index, y=mean_volatility, mode='lines', name='Mean Volatility'), row=2, col=1)

    return fig.to_json()

def optimization_graph(stocks, numPortfolios, riskFreeRate):
    stock_tickers = []
    if str(stocks) == "portfolio":
        stocks = Stock.query.all()
        stock_tickers = [stock.ticker for stock in stocks]
    else:
        stock_tickers = stocks

    if stock_tickers == [] or len(stock_tickers) == 1:
        return "1", "1", "1"

    data = get_stock_data(stock_tickers)
    data = data.dropna()

    mu = expected_returns.capm_return(data, frequency=252)
    S = risk_models.CovarianceShrinkage(data).ledoit_wolf()

    try:
        ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

        ef_max_sharpe = ef.deepcopy()
        ef_min_vol = ef.deepcopy()

        ef_max_sharpe.max_sharpe(risk_free_rate=riskFreeRate)
        ret_max_sharpe, vol_max_sharpe, _ = ef_max_sharpe.portfolio_performance(risk_free_rate=riskFreeRate)

        ef_min_vol.min_volatility()
        ret_min_vol, vol_min_vol, _ = ef_min_vol.portfolio_performance()

        n_samples = numPortfolios
        w = np.random.dirichlet(np.ones(len(mu)), n_samples)
        rets = w.dot(mu)
        vols = np.sqrt((w.dot(S).dot(w.T)).diagonal())
        sharpes = rets / vols

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vols, y=rets, mode='markers', marker=dict(size=5, color=sharpes, colorscale='Viridis'), name='Portfolios'))
        fig.add_trace(go.Scatter(x=[vol_max_sharpe], y=[ret_max_sharpe], mode='markers', marker=dict(color='red', size=10), name='Max Sharpe'))
        fig.add_trace(go.Scatter(x=[vol_min_vol], y=[ret_min_vol], mode='markers', marker=dict(color='green', size=10), name='Min Vol'))
        fig.update_layout(title_text='Efficient Frontier')
        fig.update_xaxes(title_text='Volatility')
        fig.update_yaxes(title_text='Return')

        return fig.to_json(), ef_max_sharpe.clean_weights(), ef_min_vol.clean_weights()
    
    except Exception as e:
        if str(e) == str("at least one of the assets must have an expected return exceeding the risk-free rate"):
            return "2", "2", "2"
        else:
            return "3", "3", "3"

def distribution_graph(stock):
    time_period = 30
    min_periods = 5

    stock_data = get_stock_data(stock)
    stock_close = stock_data['Close']
    stock_close_copy = stock_close.copy()
    stock_close = np.log(stock_close / stock_close.shift(1))
    stock_close = stock_close.dropna()
    stock_close.index = pd.to_datetime(stock_close.index)

    index_close = yf.download('^JKSE', start=stock_close.index.min(), end=stock_close.index.max())['Close']
    index_close = np.log(index_close / index_close.shift(1))
    index_close = index_close.dropna()
    index_close.index = pd.to_datetime(index_close.index)

    def remove_outliers(data):
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data.ffill().bfill()

        q1 = data.quantile(0.05)
        q3 = data.quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        new_data = data.where((data > lower_bound) & (data < upper_bound), np.nan)

        new_data = new_data.ffill().bfill()
        
        return pd.Series(new_data, index=data.index)

    kurtosis = stock_close.rolling(time_period).kurt()
    kurtosis = remove_outliers(kurtosis)
    mean_kurtosis = kurtosis.mean()

    skewness = stock_close.rolling(time_period).skew()
    skewness = remove_outliers(skewness)
    mean_skewness = skewness.mean()

    variance = stock_close.rolling(time_period).var()
    variance = remove_outliers(variance)
    mean_variance = variance.mean()

    mean = stock_close.rolling(time_period).mean()
    median = stock_close.rolling(time_period).median()
    mean = remove_outliers(mean)
    median = remove_outliers(median)

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=("Price", f"Kurtosis {mean_kurtosis:.2f}", f"Skewness {mean_skewness:.2f}", f"Variance {mean_variance:.2f}", "Mean and Median"))
    fig.update_layout(title_text=f"{stock[:-3].upper()} Stock Distribution Analysis, Rolling {time_period} Days", showlegend=False)

    fig.add_trace(go.Scatter(x=stock_close_copy.index, y=stock_close_copy, mode='lines', name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=kurtosis.index, y=kurtosis, mode='lines', name='Kurtosis'), row=2, col=1)
    fig.add_trace(go.Scatter(x=kurtosis.index, y=[mean_kurtosis]*len(kurtosis), mode='lines', name='Mean Kurtosis', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=kurtosis.index, y=[0]*len(kurtosis), mode='lines', name='Zero', line=dict(color='black')), row=2, col=1)

    fig.add_trace(go.Scatter(x=skewness.index, y=skewness, mode='lines', name='Skewness'), row=3, col=1)
    fig.add_trace(go.Scatter(x=skewness.index, y=[mean_skewness]*len(skewness), mode='lines', name='Mean Skewness', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Scatter(x=skewness.index, y=[0]*len(skewness), mode='lines', name='Zero', line=dict(color='black')), row=3, col=1)

    fig.add_trace(go.Scatter(x=variance.index, y=variance, mode='lines', name='Variance'), row=4, col=1)
    fig.add_trace(go.Scatter(x=variance.index, y=[mean_variance]*len(variance), mode='lines', name='Mean Variance', line=dict(color='red')), row=4, col=1)
    fig.add_trace(go.Scatter(x=variance.index, y=[0]*len(variance), mode='lines', name='Zero', line=dict(color='black')), row=4, col=1)

    fig.add_trace(go.Scatter(x=mean.index, y=mean, mode='lines', name='Mean'), row=5, col=1)
    fig.add_trace(go.Scatter(x=median.index, y=median, mode='lines', name='Median', line=dict(color='red')), row=5, col=1)

    return fig.to_json()

def ownership_graph(stock, sum_mode=False, top_n=5):
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    if not os.path.exists(dataset_dir):
        return jsonify({'error': 'Directory not found'})
    stock_file = os.path.join(dataset_dir, 'data_filtered.csv')
    if os.path.exists(stock_file):
        data = pd.read_csv(stock_file, delimiter='|')
    else:
        return jsonify({'error': 'File not found'})

    if '.JK' in stock:
        stock = stock[:-3]

    if stock not in data['Code'].unique():
        return jsonify({'error': 'Stock not found'})

    data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%Y')
    data.set_index('Date', inplace=True)
    data_filtered = data.loc[data['Code'] == stock]
    data_copy = data_filtered.copy()
    data_copy['Total Foreign'] = data_copy['Total']
    data_copy['Total Domestic'] = data_copy['Total.1']
    data_filtered = data_filtered.drop(['Code', 'Type', 'Sec. Num', 'Total', 'Total.1'], axis=1)
    data_copy = data_copy.drop(['Code', 'Type', 'Sec. Num', 'Total', 'Total.1'], axis=1)
    data_copy2 = data_filtered.copy()

    if sum_mode:
        top_holders = data_filtered.drop(['Price'], axis=1)
        top_holders = top_holders.sum().nlargest(top_n).index
        data_filtered['Others'] = data_filtered.drop(columns=top_holders).sum(axis=1)
        data_filtered = data_filtered[top_holders.append(pd.Index(['Others']))]
        data_filtered['Price'] = data_copy2['Price']
        correlation = data_filtered.corr().values
    else:
        top_holders = data_filtered.drop(['Price'], axis=1)
        data_filtered['Price'] = data_copy2['Price']
        correlation = data_filtered.corr().values

    data_filtered.index = data_filtered.index.astype(str)

    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, subplot_titles=('Type of Holders', 'Total Foreign vs Total Domestic', 'Correlation Heatmap'))
    fig.update_layout(title_text=f'{stock} Stock Holder Analysis', showlegend=False)
    fig.update_layout(barmode='stack')

    for column in data_filtered.columns:
        fig.add_trace(
            go.Bar(x=data_filtered.index, y=data_filtered[column], name=column, marker=dict(line=dict(width=0))),
            row=1, col=1
        )

    fig.add_trace(
        go.Bar(x=data_copy.index, y=data_copy['Total Foreign'], name='Foreign', marker=dict(line=dict(width=0))),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=data_copy.index, y=data_copy['Total Domestic'], name='Domestic', marker=dict(line=dict(width=0))),
        row=1, col=2
    )

    fig.add_trace(
        go.Heatmap(z=correlation, x=data_filtered.columns, y=data_filtered.columns, colorscale='Viridis'),
        row=1, col=3
    )

    return fig.to_json()

def performance_graph(stock, riskFreeRate):
    riskFreeRate = riskFreeRate / 365
    time_period = 30
    min_periods = 5

    stock_data = get_stock_data(stock)
    stock_price = stock_data['Close']
    stock_close = stock_data['Close'].pct_change().dropna()
    stock_close = stock_close.ffill().bfill()
    stock_close.index = pd.to_datetime(stock_close.index)

    index_close = yf.download('^JKSE', start=stock_close.index.min(), end=stock_close.index.max())['Close']
    index_close = index_close.pct_change().dropna()
    index_close = index_close.ffill().bfill()
    index_close.index = pd.to_datetime(index_close.index)

    def remove_outliers(data):
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        data = data.ffill().bfill()

        q1 = data.quantile(0.05)
        q3 = data.quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        new_data = data.where((data > lower_bound) & (data < upper_bound), np.nan)
        new_data = new_data.ffill().bfill()

        return pd.Series(new_data, index=data.index)

    volatility = stock_close.rolling(window=time_period, center=True, min_periods=min_periods).std() * np.sqrt(time_period)
    sharpe_ratio = (stock_close.rolling(window=time_period, center=True, min_periods=min_periods).mean() - riskFreeRate) * time_period / volatility
    sharpe_ratio = remove_outliers(sharpe_ratio)
    mean_sharpe = sharpe_ratio.mean()

    sortino_volatility = stock_close.clip(upper=0).rolling(window=time_period, center=True, min_periods=min_periods).std() * np.sqrt(time_period)
    sortino_ratio = (stock_close.rolling(window=time_period, center=True, min_periods=min_periods).mean() - riskFreeRate) * time_period / sortino_volatility
    sortino_ratio = remove_outliers(sortino_ratio)
    mean_sortino = sortino_ratio.mean()

    beta = stock_close.rolling(window=time_period, center=True, min_periods=min_periods).cov(index_close).div(index_close.rolling(window=time_period, center=True, min_periods=min_periods).var())
    expected_return = riskFreeRate + beta * (index_close.rolling(window=time_period, center=True, min_periods=min_periods).mean() - riskFreeRate)
    treynor_ratio = (expected_return - riskFreeRate) / beta
    treynor_ratio = remove_outliers(treynor_ratio)
    mean_treynor = treynor_ratio.mean()

    active_return = stock_close - index_close
    tracking_error = active_return.rolling(window=time_period, center=True, min_periods=min_periods).std() * np.sqrt(time_period)
    information_ratio = active_return.rolling(window=time_period, center=True, min_periods=min_periods).mean() / tracking_error
    information_ratio = remove_outliers(information_ratio)
    mean_information = information_ratio.mean()

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=('Price', 'Sharpe Ratio', 'Sortino Ratio', 'Treynor Ratio', 'Information Ratio'))
    fig.update_layout(title_text=f'{stock[:-3].upper()} Stock Risk-Adjusted Performance Analysis, 30 Days Period', showlegend=False)

    fig.add_trace(go.Scatter(x=stock_price.index, y=stock_price, mode='lines', name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=sharpe_ratio.index, y=sharpe_ratio, mode='lines', name='Sharpe Ratio'), row=2, col=1)
    fig.add_trace(go.Scatter(x=sharpe_ratio.index, y=[mean_sharpe]*len(sharpe_ratio), mode='lines', name='Mean Sharpe Ratio', line=dict(color='red')), row=2, col=1)

    fig.add_trace(go.Scatter(x=sortino_ratio.index, y=sortino_ratio, mode='lines', name='Sortino Ratio'), row=3, col=1)
    fig.add_trace(go.Scatter(x=sortino_ratio.index, y=[mean_sortino]*len(sortino_ratio), mode='lines', name='Mean Sortino Ratio', line=dict(color='red')), row=3, col=1)

    fig.add_trace(go.Scatter(x=treynor_ratio.index, y=treynor_ratio, mode='lines', name='Treynor Ratio'), row=4, col=1)
    fig.add_trace(go.Scatter(x=treynor_ratio.index, y=[mean_treynor]*len(treynor_ratio), mode='lines', name='Mean Treynor Ratio', line=dict(color='red')), row=4, col=1)

    fig.add_trace(go.Scatter(x=information_ratio.index, y=information_ratio, mode='lines', name='Information Ratio'), row=5, col=1)
    fig.add_trace(go.Scatter(x=information_ratio.index, y=[mean_information]*len(information_ratio), mode='lines', name='Mean Information Ratio', line=dict(color='red')), row=5, col=1)

    return fig.to_json()

def get_stock_sentiment(stock):
    titles = []

    def get_news(stock):
        stock = stock.replace('.JK', '')
        locale.setlocale(locale.LC_TIME, 'id_ID')
        
        for i in range(0, 1):
            url = f"https://www.detik.com/search/searchall?query={stock}&sortby=time&page={i}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("article")
            
            for article in articles:
                link = article.find("a")
                href = link.get("href")
                title = link.text.strip()
                date = link.text.strip()
                
                if title.startswith("detikFinance"):
                    date = re.sub(r"detik\w+,", "", date)
                    date = date.split("\n")[0].strip()
                    date = datetime.strptime(date, "%d %b %Y %H:%M WIB").strftime("%Y-%m-%d %H:%M:%S")
                    title = title.split("\n")[1].strip()
                
                if stock.upper() in title.upper():
                    if (title, date) not in titles:
                        titles.append(("Detik", title, date, href))

        for i in range(0, 2, 2):
            url = f"https://www.kontan.co.id/search/?search={stock}&per_page={i}0"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("div", class_="ket")

            pattern_days = r'(\d+)\s*Hari'
            pattern_hours = r'(\d+)\s*Jam'
            pattern_minutes = r'(\d+)\s*Menit'

            def parse_relative_time(relative_time):
                days = re.search(pattern_days, relative_time)
                hours = re.search(pattern_hours, relative_time)
                minutes = re.search(pattern_minutes, relative_time)

                days = int(days.group(1)) if days else 0
                hours = int(hours.group(1)) if hours else 0
                minutes = int(minutes.group(1)) if minutes else 0

                now = datetime.now()
                actual_time = now - timedelta(days=days, hours=hours, minutes=minutes)
                actual_time = actual_time.replace(microsecond=0)
                return actual_time

            for article in articles:
                links = article.find_all("a")
                if len(links) >= 2:
                    href = links[1].get("href")
                    if not href.startswith("https"):
                        href = "https:" + href
                
                article_text = article.text.strip().split("|")[1].strip()

                if stock.upper() in article_text.upper():
                    date_match = re.search(r'\b\d+ (?:Hari|Jam|Menit)(?: \d+ (?:Hari|Jam|Menit))* lalu\b', article_text)
                    if date_match:
                        relative_time_str = date_match.group()
                        relative_time = parse_relative_time(relative_time_str)
                        date = relative_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    title = article_text.split("  ")[1].strip()

                    if (title, date) not in titles:
                        titles.append(("Kontan", title, date, href))

        locale.setlocale(locale.LC_TIME, '')
        
        for i in range(0, 1):
            url = f"https://search.bisnis.com/?q={stock}&per_page={i}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            articles = soup.find_all("div", class_="art--row")
            
            for article in articles:
                date_text = article.find("div", class_="artDate").text.strip()

                if 'menit' in date_text or 'jam' in date_text or 'hari' in date_text:
                    number = int(date_text.split()[0])
                    unit = date_text.split()[1]

                    if 'menit' in unit:
                        delta = timedelta(minutes=number)
                    elif 'jam' in unit:
                        delta = timedelta(hours=number)
                    elif 'hari' in unit:
                        delta = timedelta(days=number)

                    actual_time = datetime.now() - delta
                    date = actual_time
                else:
                    date_text = date_text.replace(" WIB", "")
                    date = datetime.strptime(date_text, "%d %B %Y %H:%M")

                date_str = date.strftime("%Y-%m-%d %H:%M:%S") 
                
                title = article.find("h4").text.strip()
                href = article.find("a").get("href")

                if stock.upper() in title.upper():
                    if (title, date_str) not in titles:
                        titles.append(("Bisnis", title, date_str, href))

        return titles

    news = get_news(stock)
    
    sentences = []
    for title in titles:
        title = title[1]
        translated = GoogleTranslator(source='id', target='en').translate(title)
        sentences.append(translated)

    results = nlp(sentences)

    results = [result for result in results if result['score'] > 0.9]
    score = []
    final = {}
    for i, result in enumerate(results):
        final[i] = {'source': titles[i][0], 'title': titles[i][1], 'date': titles[i][2], 'link': titles[i][3], 'sentiment': result['label'], 'confidence': result['score']}
        if result['label'] == 'positive':
            score.append(1)
        elif result['label'] == 'negative':
            score.append(-1)
        else:
            score.append(0)

    return final, score

def get_stock_prediction(stock, forecast_model, forecast_period=30):
    start = datetime.now() - timedelta(days=365*5)
    end = datetime.now()
    data = yf.download(stock, start=start, end=end)

    if forecast_model == 'ARIMA':
        stock_price = data['Close'].dropna()
        stock_price = stock_price.asfreq('B').ffill()

        model = auto_arima(stock_price, seasonal=False, stepwise=True, suppress_warnings=True)
        order = model.order
        arima_model = ARIMA(stock_price, order=order)
        arima_fit = arima_model.fit()

        forecast_index = pd.date_range(start=stock_price.index[-1], periods=forecast_period + 1)[1:]
        forecast = arima_fit.forecast(steps=forecast_period)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_price.index, y=stock_price, mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=forecast_index, y=forecast, mode='lines', name='Forecast Price'))

        fig.update_layout(title_text=f'{stock[:-3]} price forecast using ARIMA for the next {forecast_period} days', showlegend=False)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')

        return fig.to_json()

    if forecast_model == 'LSTM':
        data = data[['Adj Close']].ffill()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step - 1):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        try:
            time_step = 60
            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        except IndexError:
            return "2"

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=50, epochs=5)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform([y_train])
        y_test = scaler.inverse_transform([y_test])

        future_predictions = []
        last_sequence = scaled_data[-time_step:]
        last_sequence = np.array(last_sequence).reshape(1, time_step, 1)

        for i in range(forecast_period):
            next_prediction = model.predict(last_sequence)
            future_predictions.append(next_prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_prediction

        future_predictions = scaler.inverse_transform([future_predictions])

        train_scaled_data = scaled_data[:train_size].reshape(-1, 1)
        test_scaled_data = scaled_data[train_size:].reshape(-1, 1)

        train_dates = data.index[:train_size]
        test_dates = data.index[train_size:]
        future_dates = pd.date_range(start=data.index[-1], periods=forecast_period + 1)[1:]

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=train_dates, y=scaler.inverse_transform(train_scaled_data).reshape(-1), mode='lines', name='Train Data'))
        fig.add_trace(go.Scatter(x=test_dates, y=scaler.inverse_transform(test_scaled_data).reshape(-1), mode='lines', name='Test Data'))
        fig.add_trace(go.Scatter(x=test_dates[time_step + 1:], y=test_predict.reshape(-1), mode='lines', name='Predicted Price'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions[0], mode='lines', name=f'Future {forecast_period}-day Prediction'))

        fig.update_layout(title_text=f'{stock[:-3]} price forecast using LSTM for the next {forecast_period} days', showlegend=False)
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Price')

        return fig.to_json()

    return "1"