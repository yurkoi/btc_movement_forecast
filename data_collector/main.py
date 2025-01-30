from datetime import date
import yfinance as yf
import yaml

from ipos_collector import get_ipos_data_by_years, ipos_data_preparation
from stocks_collector import stock_collect_main
from data_processing.processing import (add_technical_indicators, download_and_calculate_growth,
                                        fetch_and_process_fred_data, add_date_features,
                                        localize_and_merge, fill_missing_fields)

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def calculate_period(years_back=70, end_date=None):
    if end_date is None:
        end_date = date.today()
    start = date(year=end_date.year - years_back, month=end_date.month, day=end_date.day)
    return start, end_date


def print_period_info(start, end):
    print(f'Year = {end.year}; month = {end.month}; day = {end.day}')
    print(f'Period for indexes: {start} to {end}')


def download_and_process_data():
    """Download and process financial data."""
    return {
        "dax": download_and_calculate_growth("^GDAXI", 'dax'),
        "snp500": download_and_calculate_growth("^GSPC", 'snp500'),
        "dji": download_and_calculate_growth("^DJI", 'dji'),
        "epi_etf": download_and_calculate_growth("EPI", 'epi'),
        "vix": yf.download(tickers="^VIX", interval='1d', period='max')['Adj Close'],
        "gold": download_and_calculate_growth("GC=F", 'gold'),
        "crude_oil": download_and_calculate_growth("CL=F", 'wti_oil'),
        "brent_oil": download_and_calculate_growth("BZ=F", 'brent_oil'),
        "btc_usd": download_and_calculate_growth("BTC-USD", 'btc_usd')
    }


def fetch_and_transform_fred_data(start):
    """Fetch and transform FRED data."""
    gdppot_transformations = {
        'gdppot_us_yoy': lambda df: df['GDPPOT'] / df['GDPPOT'].shift(4) - 1,
        'gdppot_us_qoq': lambda df: df['GDPPOT'] / df['GDPPOT'].shift(1) - 1
    }
    cpilfesl_transformations = {
        'cpi_core_yoy': lambda df: df['CPILFESL'] / df['CPILFESL'].shift(12) - 1,
        'cpi_core_mom': lambda df: df['CPILFESL'] / df['CPILFESL'].shift(1) - 1
    }
    return {
        "gdppot": fetch_and_process_fred_data("GDPPOT", start, gdppot_transformations)[['gdppot_us_yoy', 'gdppot_us_qoq']],
        "cpilfesl": fetch_and_process_fred_data("CPILFESL", start, cpilfesl_transformations)[['cpi_core_yoy', 'cpi_core_mom']],
        "fedfunds": fetch_and_process_fred_data("FEDFUNDS", start),
        "dgs1": fetch_and_process_fred_data("DGS1", start),
        "dgs5": fetch_and_process_fred_data("DGS5", start),
        "dgs10": fetch_and_process_fred_data("DGS10", start)
    }


def merge_datasets(merged_df, datasets):
    """Merge multiple datasets into one."""
    for data, merge_key in datasets:
        merged_df = localize_and_merge(merged_df, data, left_on=merge_key)
    return merged_df


def process_and_save_merged_data(merged_df, output_path):
    """Fill missing fields and save the merged dataset."""
    fields_to_fill = [
        "gdppot_us_yoy", "gdppot_us_qoq", "cpi_core_yoy", "cpi_core_mom",
        "FEDFUNDS", "DGS1", "DGS5", "DGS10"
    ]
    merged_df = fill_missing_fields(merged_df, fields_to_fill)
    date = merged_df["Date"].max()
    date_str = date.strftime('%Y_%m_%d')
    merged_df.to_parquet(f'{output_path}/stocks_df_combined.parquet.brotli', compression='brotli')


if __name__ == "__main__":
    config = load_config('./data_collector/tickers.yaml')
    US_STOCKS, CRYPTOS, EU_STOCKS, INDIA_STOCKS = (
        config['stocks']['us_stocks'],
        config['cryptos'],
        config['stocks']['eu_stocks'],
        config['stocks']['india_stocks']
    )
    ALL_TICKERS = US_STOCKS + EU_STOCKS + INDIA_STOCKS + CRYPTOS

    today = date.today()

    start, end = calculate_period(years_back=70, end_date=today)
    print_period_info(start, end)

    market_data = download_and_process_data()
    fred_data = fetch_and_transform_fred_data(start)

    stocks_df = stock_collect_main(save_file=True)
    merged_df = add_technical_indicators(stocks_df, ALL_TICKERS)
    merged_df['Date'] = merged_df['Date'].dt.tz_localize(None)
    merged_df = add_date_features(merged_df)

    data_sources = [
        (market_data["dax"], "Date"),
        (market_data["snp500"], "Date"),
        (market_data["dji"], "Date"),
        (market_data["epi_etf"], "Date"),
        (fred_data["gdppot"], "Quarter"),
        (fred_data["cpilfesl"], "Month"),
        (fred_data["fedfunds"], "Month"),
        (fred_data["dgs1"], "Date"),
        (fred_data["dgs5"], "Date"),
        (fred_data["dgs10"], "Date"),
        (market_data["vix"], "Date"),
        (market_data["gold"], "Date"),
        (market_data["crude_oil"], "Date"),
        (market_data["brent_oil"], "Date"),
        (market_data["btc_usd"], "Date")
    ]

    merged_df = merge_datasets(merged_df, data_sources)

    ipo_data = get_ipos_data_by_years()
    ipo_stats = ipos_data_preparation(ipo_data)
    merged_df = localize_and_merge(merged_df, ipo_stats, left_on="Date")

    process_and_save_merged_data(merged_df, './data')