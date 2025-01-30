import yfinance as yf
import pandas as pd
import numpy as np
import time
import yaml

with open('./data_collector/tickers.yaml', 'r') as file:
    config = yaml.safe_load(file)

US_STOCKS = config['stocks']['us_stocks']
CRYPTOS = config['cryptos']
EU_STOCKS = config['stocks']['eu_stocks']
INDIA_STOCKS = config['stocks']['india_stocks']

ALL_TICKERS = US_STOCKS + EU_STOCKS + INDIA_STOCKS + CRYPTOS


def fetch_data(ticker):
    """Fetch and process data for a single ticker."""
    historyPrices = yf.download(tickers=ticker, period="max", interval="1d")

    if historyPrices.empty:
        return None

    historyPrices.columns = historyPrices.columns.get_level_values(0)
    historyPrices['Ticker'] = ticker
    historyPrices['Year'] = historyPrices.index.year
    historyPrices['Month'] = historyPrices.index.month
    historyPrices['Weekday'] = historyPrices.index.weekday
    historyPrices['Date'] = historyPrices.index.date

    for i in [1, 3, 7, 30, 90, 365]:
        historyPrices[f'growth_{i}d'] = historyPrices['Close'] / historyPrices['Close'].shift(i)

    historyPrices['growth_future_5d'] = historyPrices['Close'].shift(-5) / historyPrices['Close']
    historyPrices['growth_future_3d'] = historyPrices['Close'].shift(-3) / historyPrices['Close']
    historyPrices['growth_future_1d'] = historyPrices['Close'].shift(-1) / historyPrices['Close']

    historyPrices['SMA10'] = historyPrices['Close'].rolling(10).mean()
    historyPrices['SMA20'] = historyPrices['Close'].rolling(20).mean()
    historyPrices['growing_moving_average'] = (historyPrices['SMA10'] > historyPrices['SMA20']).astype(int)
    historyPrices['high_minus_low_relative'] = \
        (historyPrices['High'] - historyPrices['Low']) / historyPrices['Close']

    historyPrices['volatility'] = historyPrices['Close'].rolling(30).std() * np.sqrt(252)

    historyPrices['is_positive_growth_5d_future'] = (historyPrices['growth_future_5d'] > 1).astype(int)
    historyPrices['is_positive_growth_3d_future'] = (historyPrices['growth_future_3d'] > 1).astype(int)
    historyPrices['is_positive_growth_1d_future'] = (historyPrices['growth_future_1d'] > 1).astype(int)

    return historyPrices

def get_ticker_type(ticker):
    """Determine the type of the ticker."""
    if ticker in US_STOCKS:
        return 'US'
    elif ticker in EU_STOCKS:
        return 'EU'
    elif ticker in INDIA_STOCKS:
        return 'INDIA'
    else:
        return 'CRYPTO'

def stock_collect_main(save_file=True):
    stocks_data = []
    for i, ticker in enumerate(ALL_TICKERS):
        print(f"Fetching data for {i+1}/{len(ALL_TICKERS)}: {ticker}")
        data = fetch_data(ticker)
        if data is not None:
            stocks_data.append(data)
        time.sleep(2)

    stocks_df = pd.concat(stocks_data, ignore_index=True)

    stocks_df['ticker_type'] = stocks_df['Ticker'].apply(get_ticker_type)

    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'])
    stocks_df[['Open', 'High', 'Low', 'Close', 'Volume']] = \
        stocks_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float64')

    if save_file:
        stocks_df.to_parquet('./data/stocks_df.parquet.brotli', compression='brotli')

    print("Data fetching and processing complete. File saved as 'stocks.csv'.")
    return stocks_df


if __name__=="__main__":
    df = stock_collect_main(save_file=False)
    print(df.head())