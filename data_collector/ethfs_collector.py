import pprint
import time
import bs4
import pandas as pd

import undetected_chromedriver as uc
from selenium.webdriver.chrome.options import ChromiumOptions
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from datetime import date
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Scrape ETF data for a given type.")
parser.add_argument("-etf", choices=["btc", "eth"], required=True, help="Choose 'btc' for Bitcoin ETF or 'eth' for Ethereum ETF.")
args = parser.parse_args()

etf_mapping = {
    "btc": "bitcoin-etf-flow-all-data",
    "eth": "ethereum-etf-flow-all-data"
}
chosen_etf = etf_mapping[args.etf]

today = date.today()

chrome_options = ChromiumOptions()

chrome_options.add_argument('--disable-blink-features=AutomationControlled')

chrome_settings = {
    "options": chrome_options,
    "driver_executable_path": "/usr/bin/chromedriver",
    "chrome_binary_location": "/usr/bin/google-chrome"
}
caps = DesiredCapabilities().CHROME
chrome_settings["desired_capabilities"] = caps


driver = uc.Chrome(**chrome_settings)
url = f'https://farside.co.uk/{chosen_etf}/'
driver.get(url)

soup = bs4.BeautifulSoup(driver.page_source, "html.parser")
columns = [el.text for el in soup.find_all("th")]
rows_data = [el.text for el in soup.find_all("span", attrs={"class": "tabletext"})[len(columns):]]

data = list()
while rows_data:
    data.append(dict(map(lambda i, j: (i, j), columns, rows_data[:len(columns)])))
    rows_data = rows_data[len(columns):]

# Create DataFrame
df = pd.DataFrame(data)
df = df[columns]

dataframe_btc_etf = df.iloc[:-4,:]

def parse_value(x):
    x = str(x).strip()
    if x == '-':
        return np.nan
    x = x.replace(',', '')
    if x.startswith('(') and x.endswith(')'):
        x = x[1:-1]
        return -float(x)
    return float(x)

for col in dataframe_btc_etf.columns:
    if col != 'Date':
        dataframe_btc_etf[col] = dataframe_btc_etf[col].apply(parse_value)

output_file = f"../data/{chosen_etf}_{today}.csv"
dataframe_btc_etf.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
pprint.pprint(dataframe_btc_etf.tail(5))
driver.quit()
