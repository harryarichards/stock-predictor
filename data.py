from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta


def get_data(ticker: str, num_years):
    start = datetime.now() - relativedelta(years=num_years)
    data = yf.download(tickers=ticker, start=start)
    data = data.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
    data["Close Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Close"] < data["Close Tomorrow"]).astype(int)
    return data


def get_features(data, day_intervals=[2, 5, 10, 30, 45, 60, 90]):
    features = []
    for interval in day_intervals:
        # indicates how the current value relates to the rolling avg
        close_ratio = f"Close_Ratio_{interval}"
        data[close_ratio] = data["Close"] / data["Close"].rolling(interval).mean()
        # indicates how many of the previous days the price increased
        increase_days = f"Increase_Days_{interval}"
        data[increase_days] = data["Target"].shift(1).rolling(interval).sum()
        features += [close_ratio, increase_days]
    return data, features


def visualise_data(data: pd.DataFrame):
    data.plot.line(y="Close", use_index=True)
    plt.show()

    trend = data.copy()
    trend["1"] = trend["Target"]
    trend["0"] = 1 - trend["Target"]
    trend[["1", "0"]].groupby(trend.index.to_period("Y")).sum().plot.line()
    plt.show()
