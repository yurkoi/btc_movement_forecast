import pandas as pd
from io import StringIO

import plotly.express as px
import requests
import time
from datetime import datetime



def get_ipos_data_by_years(start_year=2019):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }
    current_year = datetime.now().year
    all_ipo_data = []
    print("IPOs data starting to fetch...")
    for year in range(start_year, current_year + 1):
        url = f"https://stockanalysis.com/ipos/{year}/"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            html_content = StringIO(response.text)
            ipo_dfs = pd.read_html(html_content)
            all_ipo_data.append(ipo_dfs[0])

            print(f"Year: {year}")
        else:
            print(f"Failed to fetch data for {year}")

        time.sleep(1.5)

    all_ipo_data_df = pd.concat(all_ipo_data, ignore_index=True)
    return all_ipo_data_df


def ipos_data_preparation(df):
    df['IPO Date'] = pd.to_datetime(df['IPO Date'])

    df['IPO Price'] = df['IPO Price'].apply(lambda x: x[1:] if isinstance(x, str) else x)
    df['IPO Price'] = pd.to_numeric(df['IPO Price'], errors='coerce')

    df['Current'] = df['Current'].apply(lambda x: x[1:] if isinstance(x, str) else x)
    df['Current'] = pd.to_numeric(df['Current'], errors='coerce')

    df['Return'] = df['Return'].apply(lambda x: x[:-1] if isinstance(x, str) else x)
    df['Return'] = pd.to_numeric(df['Return'].str.replace('%', ''), errors='coerce') / 100

    df['Price Increase'] = df['Current'] - df['IPO Price']

    ipo_stats = df.groupby('IPO Date').agg({
        'IPO Price': ['mean', 'median'],
        'Return': ['mean', 'median'],
        'Price Increase': ['mean', 'median']})

    ipo_stats.columns = ipo_stats.columns.get_level_values(0)
    ipo_stats.columns = ["IPO_Price_avg", "IPO_Price_med", "IPO_Return_avg", "IPO_Return_med", "IPO_Price_Increase_avg",
                         "IPO_Price_Increase_med"]

    return ipo_stats


def viz_and_save_iposbar(stacked_ipos_df):
    stacked_ipos_df['Date_monthly'] = stacked_ipos_df['IPO Date'].dt.to_period('M').dt.to_timestamp()

    monthly_deals = stacked_ipos_df['Date_monthly'].value_counts().reset_index().sort_values(by='Date_monthly')
    monthly_deals.columns = ['Date_monthly', 'Number of Deals']

    fig = px.bar(monthly_deals,
                 x='Date_monthly',
                 y='Number of Deals',
                 labels={'Month_Year': 'Month and Year', 'Number of Deals': 'Number of Deals'},
                 title='Number of IPO Deals per Month and Year',
                 text='Number of Deals')

    fig.update_traces(textposition='outside',
                      textfont=dict(color='black', size=14),)

    fig.update_layout(title_x=0.5)
    fig.write_image("ipos_monthly_deals.png")


if __name__ == "__main__":
    full_ipo_dataset = get_ipos_data_by_years()
    final_df = ipos_data_preparation(full_ipo_dataset)
    print(final_df.tail(10))
    print(final_df.info())
    final_df.to_csv('full_ipos_data.csv')
