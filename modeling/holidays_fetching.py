import holidays
import pandas as pd


def fetch_holidays(df_full):
    us_holidays = holidays.country_holidays('US')
    nyse_holidays = holidays.financial_holidays('NYSE')
    ecb_holidays = holidays.financial_holidays('ECB')

    results_us = []
    results_nyse = []
    results_ecb = []


    for date in df_full.Date:
        us_result = us_holidays.get(date)
        if us_result != None:
            results_us.append({
                'Date': date,
                'us_holiday': 1
            })

    for date in df_full.Date:
        nyse_result = nyse_holidays.get(date)
        if nyse_result != None:
            results_nyse.append({
                'Date': date,
                'nyse_holiday': 1
            })

    for date in df_full.Date:
        ecb_result = ecb_holidays.get(date)
        if ecb_result != None:
            results_ecb.append({
                'Date': date,
                'ecb_holiday': 1
            })

    results_us = pd.DataFrame(results_us)
    results_nyse = pd.DataFrame(results_nyse)
    results_ecb = pd.DataFrame(results_ecb)

    results_us.set_index('Date', inplace=True)
    results_nyse.set_index('Date', inplace=True)
    results_ecb.set_index('Date', inplace=True)


    df_full = df_full.merge(results_us, on='Date', how='left')
    df_full = df_full.merge(results_nyse, on='Date', how='left')
    df_full = df_full.merge(results_ecb, on='Date', how='left')

    df_full[['us_holiday', 'nyse_holiday', 'ecb_holiday']] = df_full[['us_holiday', 'nyse_holiday', 'ecb_holiday']].fillna(0)
    return df_full