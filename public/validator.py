import yfinance as yf
import pandas as pd
import os

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
                    print(e)
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

def validate_stock(ticker):
    """
    This function validates a stock ticker and modifies it if necessary.
    """

    ticker = ticker.upper().strip()

    if not ticker.isalpha():
        return False, ticker

    if len(ticker) != 4:
        return False, ticker

    if not ticker.endswith('.JK'):
        ticker = ticker + '.JK'

    try:
        stock_data = get_stock_data(ticker)
        if stock_data.empty:
            return False, ticker
    except ValueError:
        return False, ticker

    return True, ticker