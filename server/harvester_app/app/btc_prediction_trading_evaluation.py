from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
import time
from binance.exceptions import BinanceAPIException, BinanceOrderException
#import copy
from matplotlib.dates import date2num
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle

coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ"#, "ERG"
    ]


mins = [
    15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
    869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230# , 1.1
    ]
tops = [
    69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
    4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85 #, 19.23
    ]

mins_dict = dict(zip(coins, mins))
tops_dict = dict(zip(coins, tops))

def norm_augment(df) :
    #for the gradient d1 and d2
    x = np.array([date2num(d) for d in  df.index])
    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    rsi = bl.rsi(df, period = 8)
    df = df.join([macd.df, rsi.df])
    #df.loc[:,'histogram'] =[0 if abs(x) < 1e-06 else x for x in  df['histogram']] # reducing the noize 
    df.loc[:,'h_2'] = df.loc[:,'histogram'].shift(periods=1)
    df.loc[:,'h_3'] = df.loc[:,'histogram'].shift(periods=2)
    df.loc[:,'h_4'] = df.loc[:,'histogram'].shift(periods=3)
    df.loc[:,'h_5'] = df.loc[:,'histogram'].shift(periods=4)
    df.loc[:, 'c_2'] = df.loc[:,'close'].shift(periods=1)
    df.loc[:, 'o_2'] = df.loc[:,'open'].shift(periods=1)
    df.loc[:, 'hi_2'] = df.loc[:,'high'].shift(periods=1)
    df.loc[:, 'l_2'] = df.loc[:,'low'].shift(periods=1)
    df.loc[:, 'c_3'] = df.loc[:,'close'].shift(periods=2)
    df.loc[:, 'c_4'] = df.loc[:,'close'].shift(periods=3)
    df.loc[:, 'c_5'] = df.loc[:,'close'].shift(periods=4)
    df.loc[:, 'c_6'] = df.loc[:,'close'].shift(periods=5)

    y = df.loc[:,'histogram'].values
    df.loc[:, "d1"] = np.gradient(y, x)
    df.loc[:, "d2"] = np.gradient(df.loc[:, "d1"].values, x)

    y = df.loc[:,'close'].values
    df.loc[:, "c1"] = np.gradient(y, x)
    df.loc[:, "c2"] = np.gradient(df.loc[:, "c1"].values, x)
    
    y = df.loc[:,'rsi'].values
    df.loc[:, "r1"] = np.gradient(y, x)
    df.loc[:, "r2"] = np.gradient(df.loc[:, "r1"].values, x)
    
    df.dropna(inplace = True)
    return df

def norm_augment_OR (df,coin, mins = mins_dict, tops = tops_dict) :
    rng = range(len(df.index))
    top_price = tops[coin]
    botom_price = mins[coin]
    #first : store the price in the avg col 
    df.loc[:,'avg'] = df.loc[:,"close"].copy()#[(df["open"][i] + df["close"][i])/2 for i in rng]

    # !!! SUPER IMPORTANT price values are overwritten with their normalized values (min - max normalisation)
    # this step makes it possible to compare the MACD values across all coins 
    # MACD values are not calculated on the absolute prices but on their respective normalized values
    df.loc[:,"open"] = [(df["open"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"high"] = [(df["high"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"low"] = [(df["low"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"close"] = [(df["close"][i] - botom_price)/(top_price - botom_price) for i in rng]

    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    df = df.join(macd.df)
    df.dropna(inplace = True)
    #df.loc[:,'histogram'] =[0 if abs(x) < 1e-03 else x for x in  df['histogram']] # reducing the noize
    x = np.array([date2num(d) for d in  df.index]) 
    y = df.loc[:,'histogram'].values
    df.loc[:, "d1"] = np.gradient(y, x)
    df.loc[:, "d2"] = np.gradient(df.loc[:, "d1"].values, x)
    return df

def predict (series, pair):
    #add proba
    print("making predition now")
    with open(f'/home/honeybadger/projects/harvester/models/{pair}_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    # Use the loaded model to make predictions
    prediction =  model.predict_proba(series)
    probas = [p[0] for p in prediction]
    
    return probas

def update_pair (pair):
    cli = unlock()
    bars = cli.get_historical_klines(pair, "1h", timestamp)
    #create empty dataframe
    df = pd.DataFrame( columns= ['date', 'open', 'high', 'low', 'close'])
    for bar in bars:
        # add each row to the dataframe
        df.loc[len(df.index)] = [float(x) for x in bar[0:5]]
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def unlock (fname = '/home/honeybadger/projects/harvester/nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    #df.set_index('date', inplace=True)
    #df.index = pd.to_datetime(df.index, unit='ms')
    return df

def d1d2_predict(row):
    d1 = row["d1"]
    d2 = row["d2"]
    x = 1 if (d1 > 0.0003 and d2 >= 0)  else 0 
    return x

timestamp =  "1681815600000"
    # This part is useless if data in c_dict is up to date
new_df = update_pair('BTCBUSD')

augmented_df = norm_augment(new_df)
candle = augmented_df["open"] - augmented_df["close"]
y_test = [0 if c < 0 else 1 for c in candle]
last_n_points = len(y_test)


probas = predict(augmented_df, pair="BTCBUSD")
y_pred_xgb = [0 if x < 0.5 else 1 for x in probas]


c_data = import_coin_data("/home/honeybadger/projects/harvester/data/h/BTCBUSD.csv")  #from old csvs !!! ALSO ADDS DATES AS INDEXES
full_data = pd.concat([c_data, new_df], axis=0)
full_data = norm_augment_OR(full_data, "BTC")
y_pred_d1d2macd = full_data.apply(d1d2_predict, axis=1)
last_n_y = y_pred_d1d2macd[-last_n_points : ]

print('Accuracy:', accuracy_score(y_test, y_pred_xgb))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred_xgb))
print('Classification report:\n', classification_report(y_test, y_pred_xgb))


print('Accuracy d1d2:', accuracy_score(y_test, last_n_y))
print('Confusion matrix d1d2:\n', confusion_matrix(y_test, last_n_y))
print('Classification report d1d2:\n', classification_report(y_test, last_n_y))
print(sum(last_n_y))
