from tqdm import tqdm
import requests
import pandas as pd
import time


def historical_tvl():
    list_of_chains = [
        'Ethereum', 'Arbitrum', 'Optimism',  'Binance', 'Avalanche', 'Polygon',
         'Kucoin', 'Base']

    all_data = []

    for blockchain in tqdm(list_of_chains):
        response = requests.get(f'https://api.llama.fi/v2/historicalChainTvl/{blockchain}')

        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df.columns = ['Date', f'{blockchain}_tvl']
            all_data.append(df)
        else:
            print(f"Failed to fetch data for {blockchain}")

        time.sleep(1.5)

    if all_data:
        result_df1 = all_data[0]
        for df in all_data[1:]:
            result_df1 = pd.merge(result_df1, df, on='Date', how='outer')

        result_df1.sort_values(by='Date', inplace=True)
        result_df1.reset_index(drop=True, inplace=True)
    else:
        print("No data was fetched successfully.")

    return result_df1


def stable_coins_charts():
    list_of_chains = [
        'Ethereum', 'Arbitrum', 'Optimism', 'Binance', 'Avalanche', 'Polygon',
        'Kucoin', 'Base'
    ]

    all_data = []

    for blockchain in tqdm(list_of_chains):
        response = requests.get(f'https://stablecoins.llama.fi/stablecoincharts/{blockchain}')

        if response.status_code == 200:
            data = response.json()

            parsed_data = []
            for entry in data:
                total_circulating = entry.get('totalCirculating', {})
                parsed_data.append({
                    'Date': pd.to_datetime(entry['date'], unit='s'),
                    f'{blockchain}_CirculatingUSD': total_circulating.get('peggedUSD', 0),
                    f'{blockchain}_CirculatingEUR': total_circulating.get('peggedEUR', 0)
                })

            df = pd.DataFrame(parsed_data)
            all_data.append(df)
            print(f"Data fetched successfully for {blockchain}")
        else:
            print(f"Failed to fetch data for {blockchain}")

        time.sleep(1.5)

    if all_data:
        result_df = all_data[0]
        for df in all_data[1:]:
            result_df = pd.merge(result_df, df, on='Date', how='outer')

        result_df.sort_values(by='Date', inplace=True)
        result_df.reset_index(drop=True, inplace=True)
    else:
        print("No data was fetched successfully.")

    return result_df


