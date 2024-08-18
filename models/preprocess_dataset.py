import pandas as pd
import talib


stocks_df = pd.read_csv("nee-stocks.csv", sep=",")
stocks_df = stocks_df.drop(columns="Symbol")

# creating new columns with shifted information
stocks_df['5d_close_pct'] = stocks_df['Close'].pct_change(5)
stocks_df['5d_future_close'] = stocks_df['Close'].shift(5)
stocks_df['5d_future_close_pct'] = stocks_df['5d_future_close'].pct_change(5, fill_method=None)

# creating new columns with some magical statistics by talib
# talib functions require numpy arrays, not pandas series, so this is why .values is used
# smooth-moving-average
stocks_df["ma14"] = talib.SMA(stocks_df['Close'].values, timeperiod=14)
# relative-strength-index; if high > the prize is expected to decrease and reversly; values between 0 and 100
stocks_df["rsi14"] = talib.RSI(stocks_df['Close'].values, timeperiod=14)
stocks_df["ma3"] = talib.SMA(stocks_df['Close'].values, timeperiod=3)
stocks_df["rsi3"] = talib.RSI(stocks_df['Close'].values, timeperiod=3)

stocks_df = stocks_df.dropna()

#save it again to csv, so that you don't need any more deal with talib library
stocks_df.to_csv("nee_stocks_with_features.csv")
