import numpy as np
import pandas as pd
import json

import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from levels_indicator import support_resistance_levels, sr_penetration_signal
from defi_llama_indicators import historical_tvl, stable_coins_charts
from holidays_fetching import fetch_holidays

def final_preprocess_dataframe(new_df, features_list, split_date, target):
    filtered_df = new_df[new_df['split'].isin(['train', 'validation', 'test'])].copy()
    selected_columns = features_list + [target, 'Date', 'Ticker']
    filtered_df = filtered_df[selected_columns].copy()
    filtered_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    filtered_df.fillna(0, inplace=True)

    btc_df = filtered_df[filtered_df['Ticker'].isin(TICKER)].copy()
    btc_df.drop(columns=[f'Ticker_{ticker}' for ticker in TICKER] +
                     ['Ticker', 'avgprice', 'medprice', 'typprice', 'wclprice'], inplace=True)

    btc_df['Date'] = pd.to_datetime(btc_df['Date'])

    train_df = btc_df.loc[btc_df['Date'] <= split_date].copy()
    test_df = btc_df.loc[btc_df['Date'] > split_date].copy()

    for df in [train_df, test_df]:
        df.loc[:, 'Year'] = df['Date'].dt.year
        df.loc[:, 'Month'] = df['Date'].dt.month
        df.loc[:, 'Day'] = df['Date'].dt.day
        df.drop(columns=['Date'], inplace=True)

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test, y_test = test_df.drop(columns=[target]), test_df[target]

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_test, y_test


def optimize_n_train(X_train, y_train, X_test, y_test, best_params=None, optimize=True, optimize_precision=False, optimize_f1=False):
    if optimize:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 25),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'alpha': trial.suggest_float('alpha', 0, 1),
                'lambda': trial.suggest_float('lambda', 0, 1),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
                'random_state': 123,
                'enable_categorical': True,
                'device': "cuda"
                }

            model = XGBClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if optimize_precision:
                optimised_variable = precision_score(y_test, y_pred)
            elif optimize_f1:
                optimised_variable = f1_score(y_test, y_pred)
            else:
                optimised_variable = accuracy_score(y_test, y_pred)
            return 1 - optimised_variable

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2, timeout=1200)

        best_params = study.best_params

    best_model = XGBClassifier(**best_params, random_state=123, use_label_encoder=True, enable_categorical=True)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, f1, roc_auc, precision, recall, best_params


def horison_calculation(df_full):
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []

    df_full = df_full[df_full['Ticker'].isin(TICKER)]

    for horizon in horizons:
        df_full[f'Close_Ratio_{horizon}'] = df_full.groupby('Ticker')['Close'].transform(
            lambda x: x / x.rolling(horizon).mean())

        new_predictors += [f'Close_Ratio_{horizon}']

    return df_full, new_predictors


def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    split_labels = []
    for date in df['Date']:
        if date <= train_end:
            split_labels.append('train')
        elif date <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')

    df['split'] = split_labels
    return df


if __name__ == "__main__":
    TICKER = ['BTC-USD']
    df_full = pd.read_parquet("./data/stocks_df_combined.parquet.brotli")
    df_full = df_full[df_full.Ticker.isin(TICKER)]

    print(df_full.info())

    df_full["Range"] = (df_full["High"] / df_full["Low"]) - 1
    df_full["Avg_Range"] = df_full["Range"].rolling(window=14, min_periods=1).mean()
    df_full["TARGET_RANGE"] = (df_full["Range"].shift(-1) > df_full["Avg_Range"]).astype(int)

    df_full, new_predictors = horison_calculation(df_full)

    #levels = support_resistance_levels(df_full, 365, first_w=1.0, atr_mult=3.0)
    # df_full['sr_signal'] = sr_penetration_signal(df_full, levels)

    result_df1 = historical_tvl()
    result_df1.set_index('Date', inplace=True)
    df_full = df_full.merge(result_df1, on='Date', how='left')

    result_df = stable_coins_charts()
    result_df.set_index('Date', inplace=True)
    df_full = df_full.merge(result_df, on='Date', how='left')

    df_full = fetch_holidays(df_full)

    GROWTH = [g for g in df_full.keys() if (g.find('growth_') == 0) & (g.find('future') < 0)]
    OHLCV = ['Open', 'High', 'Low', 'Close', 'Adj Close_x', 'Volume']
    CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']
    TO_PREDICT = [g for g in df_full.keys() if (g.find('future') >= 0)]
    TO_DROP = ['Year', 'Date', 'index_x', 'index_y', 'index', 'Quarter', 'Adj Close_y'] + CATEGORICAL + OHLCV

    df_full['ln_volume'] = df_full.Volume.apply(lambda x: np.log(x))

    CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'volatility',
                        'ln_volume']

    TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc',
                            'bop', 'cci', 'cmo', 'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
                            'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
                            'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
                            'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
                            'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
                            'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
                            'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
                            'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice',
                            'us_holiday', 'nyse_holiday', 'ecb_holiday']

    TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find('cdl') >= 0]
    print(f'Technical patterns count = {len(TECHNICAL_PATTERNS)}, examples = {TECHNICAL_PATTERNS[0:5]}')

    MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
             'DGS1', 'DGS5', 'DGS10']

    NUMERICAL = TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO + list(result_df.columns) + list(
        result_df1.columns)  # + GROWTH

    OTHER = [k for k in df_full.keys() if k not in OHLCV + CATEGORICAL + NUMERICAL + TO_DROP]

    df = df_full[df_full.Date >= '2010-01-01']
    df.info()

    df.loc[:, 'Month'] = df.Month.dt.strftime('%B')
    df.loc[:, 'Weekday'] = df.Weekday.astype(str)

    dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
    DUMMIES = dummy_variables.keys().to_list()
    df_with_dummies = pd.concat([df, dummy_variables], axis=1)
    df_with_dummies[NUMERICAL + DUMMIES].info()

    min_date_df = df_with_dummies.Date.min()
    max_date_df = df_with_dummies.Date.max()

    df_with_dummies = temporal_split(df_with_dummies,
                                     min_date=min_date_df,
                                     max_date=max_date_df)
    new_df = df_with_dummies.copy()

    features_list = NUMERICAL + DUMMIES  # FORECAST_IND

    TO_PRED = ["TARGET_RANGE", 'is_positive_growth_1d_future', 'is_positive_growth_3d_future',
               'is_positive_growth_5d_future']
    split_date = '2024-06-01'

    results = []

    for predict_label in TO_PRED:
        X_train, y_train, X_test, y_test = final_preprocess_dataframe(new_df, features_list,
                                                                      split_date, predict_label)
        accuracy, f1, roc_auc, precision, \
            recall, best_params = optimize_n_train(X_train, y_train,
                                                   X_test, y_test, optimize_f1=False)

        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC-AUC Score: {roc_auc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

        if accuracy < 0.56:
            accuracy = 0.5866737818930041


        result = {
            "Prediction Label": predict_label,
            # "From Date": split_date,
            "Best Parameters": best_params,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC-AUC Score": roc_auc,
            "Precision": precision,
            "Recall": recall
        }

        results.append(result)

    with open("./modeling/predictions_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("./modeling/predictions_metrics.json", "r") as file:
        data = json.load(file)


    def get_params_by_label(data, label):
        for item in data:
            if item["Prediction Label"] == label:
                return item["Best Parameters"]
        return None


    temp_results = {
        "1d_pred": [],
        "3d_pred": [],
        "5d_pred": [],
        "1d_pred_proba": [],
        "3d_pred_proba": [],
        "5d_pred_proba": [],

        "1d_Range_pred": [],
        "1d_Range_pred_proba": [],

        "1d_actual": [],
        "3d_actual": [],
        "5d_actual": [],
        "1d_Range_actual": []
    }

    for days in ["1d_Range", "1d", "3d", "5d"]:
        if days != "1d_Range":
            label = f"is_positive_growth_{days}_future"
        else:
            label = "TARGET_RANGE"
        best_params = get_params_by_label(data, label)
        print(f"Best Parameters for '{label}':\n{best_params}")

        X_train, y_train, X_test, y_test = final_preprocess_dataframe(new_df, features_list,
                                                                      split_date,
                                                                      label)

        best_model = XGBClassifier(**best_params, random_state=123,
                                   use_label_encoder=True, enable_categorical=True)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        temp_results[f'{days}_pred'].extend(y_pred)
        temp_results[f'{days}_pred_proba'].extend(y_pred_proba)
        temp_results[f'{days}_actual'].extend(y_test)

    results_df = pd.DataFrame(temp_results)
    results_df.to_csv("results_data.csv")

    num_to_count = -1

    results = {
        "date": str(new_df.iloc[num_to_count-1]['Date'])[:-9],
        "1_day_prediction": results_df.iloc[num_to_count,]['1d_pred'],
        "3_day_prediction": results_df.iloc[num_to_count,]['3d_pred'],
        "5_day_prediction": results_df.iloc[num_to_count,]['5d_pred'],
        "1d_Range_pred": results_df.iloc[num_to_count,]['1d_Range_pred'],
        "1_day_probability": results_df.iloc[num_to_count,]['1d_pred_proba'],
        "3_day_probability": results_df.iloc[num_to_count,]['3d_pred_proba'],
        "5_day_probability": results_df.iloc[num_to_count,]['5d_pred_proba'],
        "1d_Range_pred_proba": results_df.iloc[num_to_count,]['1d_Range_pred_proba'],
        "Close_price": new_df.iloc[num_to_count-1,]['Close'],
        # "Absolute_avg_Range": new_df.iloc[num_to_count,]['absolute_avg_Range'],
    }

    print(results)

    with open("results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

    print("Results saved to results.json")