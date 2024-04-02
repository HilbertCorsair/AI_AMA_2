from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
import time
import datetime
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pickle




def norm_augment(df) :
    #for the gradient d1 and d2
    x = np.array([date2num(d) for d in  df.index])
    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    rsi = bl.rsi(df, period = 8)
    df = df.join([macd.df, rsi.df])
    # Adding the indicators : shifting positive brings the data foreward from the past to present observation
    #-------- Last candle is from an hour ago, shift 1 is from 2 h ago ...etc
    #---------last 9 candles------
    df.loc[:, "movement"] = [1 if df.loc[i, "open"] < df.loc[i, "close"] else 0 for i in df.index]
    df.loc[:, "past_2_movement"] = (df.loc[:, "movement"].shift(periods=1))
    df.loc[:, "past_3_movement"] = (df.loc[:, "movement"].shift(periods=2))
    df.loc[:, "past_4_movement"] = (df.loc[:, "movement"].shift(periods=3))
    df.loc[:, "past_5_movement"] = (df.loc[:, "movement"].shift(periods=4))
    df.loc[:, "past_6_movement"] = (df.loc[:, "movement"].shift(periods=5))
    df.loc[:, "past_7_movement"] = (df.loc[:, "movement"].shift(periods=6))
    df.loc[:, "past_8_movement"] = (df.loc[:, "movement"].shift(periods=7))
    df.loc[:, "past_9_movement"] = (df.loc[:, "movement"].shift(periods=8))
    #----------------------------------------------------------------------- 
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
with open(f'/home/honeybadger/projects/harvester/models/ADABUSD_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)

def predict (series):
    #add proba
    #print("making predition now")
    
    # Use the loaded model to make predictions
    prediction =  model.predict_proba(series) 
    probas = [p[0] for p in prediction]
    return probas

def update_pair (pair, timestamp):
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


old_data = import_coin_data('/home/honeybadger/projects/harvester/data/h/ADABUSD.csv')
test_start_index = old_data.index[-1]
timestamp = int(old_data.index[-1].timestamp()) * 1000
#new_data = update_pair('ADABUSD', timestamp)
#new_data.to_pickle(path = "/home/honeybadger/projects/harvester/data/h/pkls/f_test_17092023_ADABUSD.pkl")
new_data = pd.read_pickle("/home/honeybadger/projects/harvester/data/h/pkls/f_test_17092023_ADABUSD.pkl")
new_data = new_data.loc[new_data.index[1] : ] #removing the first obs since it is the same as the last obs of old data. 
#nbr_of_values = len(new_data.index)
print(len(new_data.index))
"""
P = []
while not new_data.empty:
    print(f'Srating : {new_data.shape[0]} rows')
    #take the index or the first row
    ind = new_data.index[0]
    # Take the first row from new data 
    row = new_data.loc[ind]
    
    # Append the row to B
    old_data = old_data.append(row)

    #Norm and augment 
    augmented_data = norm_augment(old_data)
   
    row = augmented_data.iloc[-1].values.reshape(1, -1)

    #make a prediction
    p = predict(row)
    print(p)
    P.append(p)
    # Drop the first row from A
    new_data = new_data.drop(ind)
"""
P = pd.read_pickle("/home/honeybadger/projects/harvester/data/h/pkls/f_test_P_ADABUSD.pkl") 
P = P.values
P =  np.array([item[0] for item in P])
P = P[: -1]

#pd.Series(P).to_pickle(path = "/home/honeybadger/projects/harvester/data/h/pkls/f_test_P_ADABUSD.pkl")
new_data.loc[:, "movement"] = [1 if new_data.loc[i, "open"] < new_data.loc[i, "close"] else 0 for i in new_data.index]
new_data["target"] =  new_data.loc[:, "movement"].shift(periods=-1)
predictions = [0 if x >=0.5 else 1 for x in P]
target = new_data["target"].values

target = target[: -1]

conf_mat = confusion_matrix(target, predictions)
print(conf_mat)
    

