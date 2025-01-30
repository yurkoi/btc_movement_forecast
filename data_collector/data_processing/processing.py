from .tech_indicators import (talib_get_pattern_recognition_indicators,
                             talib_get_volume_volatility_cycle_price_indicators,
                             talib_get_momentum_indicators_for_one_ticker)
import pandas as pd
import pandas_datareader as pdr
import time
import yfinance as yf
import yaml


def add_technical_indicators(stocks_df, tickers, verbose=True):
    merged_df_with_tech_ind = pd.DataFrame()

    i = 0
    for ticker in tickers:
      i += 1
      if verbose:
        print(f'{i}/{len(tickers)}: Processing ticker {ticker}')
      current_ticker_data = stocks_df[stocks_df['Ticker'] == ticker].copy()
      current_ticker_data['Date'] = pd.to_datetime(current_ticker_data['Date'], utc=True)

      df_momentum = talib_get_momentum_indicators_for_one_ticker(current_ticker_data)
      df_momentum["Date"] = pd.to_datetime(df_momentum['Date'], utc=True)

      df_volume = talib_get_volume_volatility_cycle_price_indicators(current_ticker_data)
      df_volume["Date"] = pd.to_datetime(df_volume['Date'], utc=True)

      df_pattern = talib_get_pattern_recognition_indicators(current_ticker_data)
      df_pattern["Date"] = pd.to_datetime(df_pattern['Date'], utc=True)

      m1 = pd.merge(current_ticker_data, df_momentum.reset_index(), how='left',
                    on=["Date", "Ticker"], validate="one_to_one")
      m2 = pd.merge(m1, df_volume.reset_index(), how='left', on=["Date", "Ticker"], validate="one_to_one")
      m3 = pd.merge(m2, df_pattern.reset_index(), how='left', on=["Date", "Ticker"], validate="one_to_one")

      if merged_df_with_tech_ind.empty:
        merged_df_with_tech_ind = m3
      else:
        merged_df_with_tech_ind = pd.concat([merged_df_with_tech_ind, m3], ignore_index=True)

    merged_df_with_tech_ind['Date'] = pd.to_datetime(merged_df_with_tech_ind['Date']).dt.tz_localize(None)
    return merged_df_with_tech_ind


def download_and_calculate_growth(ticker, prefix, interval='1d', period='max', delay=2):
    df = yf.download(tickers=ticker, interval=interval, period=period)
    print(f"Fetching data for: {ticker}")
    time.sleep(delay)
    if df.empty:
      return pd.DataFrame()  # Return empty DataFrame if no data is fetched
    for i in [1, 3, 7, 30, 90, 365]:
      df['growth_' + prefix + '_' + str(i) + 'd'] = df['Close'] / df['Close'].shift(i)
    growth_keys = [k for k in df.keys() if k[0].startswith('growth')]
    df = df[growth_keys]
    df.columns = df.columns.get_level_values(0)
    return df


def fetch_and_process_fred_data(series_id, start, transformations=None):
    print(f"Fetching data for {series_id}.")
    data = pdr.DataReader(series_id, "fred", start=start)
    if transformations:
      for col_name, transform in transformations.items():
        data[col_name] = transform(data)
    time.sleep(1)
    return data


def localize_and_merge(base_df, merge_df, left_on, right_index=True, time_zone=None, validate="many_to_one"):
    if time_zone:
      merge_df.index = merge_df.index.tz_localize(time_zone)
    else:
      merge_df.index = merge_df.index.tz_localize(None)

    return pd.merge(
      base_df, merge_df, how="left", left_on=left_on, right_index=right_index, validate=validate)


def add_date_features(df):
    df["Quarter"] = df["Date"].dt.to_period("Q").dt.to_timestamp()
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df


def fill_missing_fields(df, fields):
    for field in fields:
      df[field] = df[field].ffill()
    return df


if __name__ == "__main__":
    with open('tickers.yaml', 'r') as file:
      config = yaml.safe_load(file)

    us_stocks = config['stocks']['us_stocks']
    cryptos = config['cryptos']
    eu_stocks = config['stocks']['eu_stocks']
    india_stocks = config['stocks']['india_stocks']

    ALL_TICKERS = us_stocks + eu_stocks + india_stocks + cryptos

    stocks_df = pd.read_csv("../../data/stocks.csv")

    merged_data = add_technical_indicators(stocks_df, ALL_TICKERS)
    print(merged_data.tail())