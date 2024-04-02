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

of_interest = ["BTC", "MINA",  "AGIX", "ALGO", "MATIC", "KAVA", "FIL", "RNDR", "ADA", "BNB", "AR"]
#of_interest = ["BTC"]
pairs = [f'{coin}BUSD' for coin in of_interest]
BUSD_decs = [2, 3,  5, 4, 4, 3, 3, 3, 4, 4, 3]
C_decs = [5, 1, 0, 0, 1, 1, 2, 2, 1, 3, 2]

# First import data and process it
#BUSD_decs = [2, 4, 3, 0, 5, 3, 4, 1, 4, 3, 2, 1, 2]
rounding_order_price = dict(zip(pairs, BUSD_decs))
# CRYPRO AMOUNT 
#C_decs = [5, 1, 1, 4, 0, 2, 0, 3, 1, 2, 2, 3, 2]
rounding_order_crypro_amount = dict(zip(pairs, C_decs))

def norm_augment(df) :
    #for the gradient d1 and d2
    x = np.array([date2num(d) for d in  df.index])
    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    rsi = bl.rsi(df, period = 8)
    df = df.join([macd.df, rsi.df])
    # Adding the indicators : shifting brings the data foreward from the past to present observation
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


def predict (series, pair = "ADABUSD"):
    #add proba
    #print("making predition now")
    with open(f'/home/honeybadger/projects/harvester/models/{pair}_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    # Use the loaded model to make predictions
   
    prediction =  model.predict_proba(series)
    
    probas = [p[0] for p in prediction] # !!! for some strange reason the probabolities seem to work best when clsses are inverted (0 = p of green)
    
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

data_files =  [f'/home/honeybadger/projects/harvester/data/h/{c}BUSD.csv'for c in of_interest]

#c_data = [import_coin_data(pth) for pth in data_files] #from old csvs !!! ALSO ADDS DATES AS INDEXES 
#c_data = [pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl") for p in pairs]

old_ada_data = import_coin_data('/home/honeybadger/projects/harvester/data/h/ADABUSD.csv')
timestamp =  "1681812000000"
new_ada_data = update_pair("ADABUSD")

print(old_ada_data.tail(3))
print(new_ada_data.tail(3))

n_obs = len(new_ada_data.index)
for n in n_obs:
    old_ada_data =  pd.concat([old_ada_data, new_ada_data.iloc[n, : ]], axis=0)

exit()

#all_ada =  pd.concat([old_ada_data, new_ada_data], axis=0)
all_ada = norm_augment(old_ada_data) 

test_ind = new_ada_data.index[0]
y_test = all_ada.loc[test_ind: , "movement"].shift(periods= -1).values[0:-1]
y_pred = [0 if x > 0.5 else 1 for x in predict(all_ada.loc[test_ind:,  : ] )][0:-1]

y_pred2 = [predict(series=series) for series in [all_ada.loc[i:i, : ] for i in new_ada_data.index ]]

y_pred2 = [0 if x[0] > 0.5 else 1 for x in y_pred2  ]


print('Accuracy:', accuracy_score(y_test, y_pred2[0:-1]))
print('Confusion matrix:\n', confusion_matrix(y_test, y_pred2[0:-1]))
print('Classification report:\n', classification_report(y_test, y_pred2[0:-1]))

#c_dict = dict(zip(pairs, c_data))
#timestamp =  str(int(c_dict["ADABUSD"].index[-1].timestamp()*1000))
    # This part is useless if data in c_dict is up to date
""" 
for p in pairs:
    
    # This part is useless if data in c_dict is up to date
    new_df = update_pair(p)
    if not new_df.empty:
        # Add data to existing dict and also save updatated to pkl
        c_dict[p] = pd.concat([c_dict[p], new_df], axis=0)
        c_dict[p].to_pickle(path = f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl")
    else:
        print("No new data.")
print("Done with the files")
"""

