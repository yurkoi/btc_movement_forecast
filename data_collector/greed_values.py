import requests
import json
from io import StringIO
import pandas as pd

url = "https://api.alternative.me/fng/?limit=0&format=json"

response = requests.get(url)

if response.status_code == 200:
    csv_data = response.text
    data = json.loads(csv_data)
    df = pd.DataFrame(data['data'])
    df.drop(['time_until_update'], axis=1, inplace=True)
    print(df.head(10))
else:
    print(f"Request failed with status code {response.status_code}")


df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
df = df.iloc[::-1].reset_index(drop=True)
df = pd.get_dummies(df, columns=['value_classification'])
df = df.replace({True: 1, False: 0})
df['14_day_mean_greed'] = df['value'].rolling(window=14).mean()
df['14_day_std_greed'] = df['value'].rolling(window=14).std()
df.rename(columns={"value": "greed_value", "timestamp": "Datetime"}, inplace=True)

df.to_csv("../data/greed_values.csv")