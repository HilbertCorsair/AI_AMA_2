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


def predict (series, pair = "ADABUSD"):
    #add proba
    #print("making predition now")
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

def get_price(pair):
    cli = unlock()
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, rounding_order_price[pair])
    return price

def get_ada_busd_ballance ():
    cli = unlock()
    ada_price = get_price(f'ADABUSD')
    balance = [cli.get_asset_balance(c)for c in ["BUSD", "ADA"] ]
    stash = dict(zip(["BUSD", "ADA"], [float(d["free"]) for d in balance]))
    in_ada = stash["ADA"] * ada_price
    total = in_ada + stash['BUSD']
    busd = stash["BUSD"]/(stash['BUSD'] + in_ada)
    ada = in_ada/total
    slack = 10 /total
    
    return (busd, ada, slack, total, ada_price)

def sell( pair, q , price):
    cli = unlock()
    try: 
        cli.create_order(
            symbol = pair,
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        print(f"Placing SELL {pair} order ...")
        #open_order = cli.get_open_orders(symbol = pair )
        while cli.get_open_orders(symbol = pair ):
            #orderID = open_order[0]['orderId']
            if c1 < 1 :
                print("waiting for order to be filled ...")
                c1 += 1
        
            else : 
                c1 += 1
                time.sleep(30)

        #print("DONE!")

    except BinanceAPIException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
def buy( pair, q , price):
    cli = unlock()
    try: 
        cli.create_order(
            symbol = pair,
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        print(f'Placeing BUY order...')
        #open_order = cli.get_open_orders(symbol = pair )
        while cli.get_open_orders(symbol = pair ):
            #orderID = open_order[0]['orderId']
            if c1 < 1 :
                print("waiting for order to be filled ...")
                c1 += 1
            else :
                c1 += 1
                time.sleep(30)
        #print("DONE!")

    except BinanceAPIException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e) 

pairs = ["ADABUSD"]
BUSD_decs = [4]
C_decs = [1]
rounding_order_price = dict(zip(pairs, BUSD_decs))
rounding_order_crypro_amount = dict(zip(pairs, C_decs))

#past_data = import_coin_data('/home/honeybadger/projects/harvester/data/h/ADABUSD.csv')

past_data = pd.read_pickle("/home/honeybadger/projects/harvester/data/h/pkls/ADABUSD.pkl")
timestamp =  str((past_data.index[-2].timestamp())*1000) # timestamp in milisec

f_test_start_date = datetime.datetime.fromtimestamp((past_data.index[-2].timestamp())).strftime('%Y-%m-%d %H:%M')

"""
new_df = update_pair("ADABUSD")
 
if not new_df.empty:
    # Add data to existing dict and also save updatated to pkl
    past_data = pd.concat([past_data, new_df], axis=0)
    past_data.to_pickle(path = f"/home/honeybadger/projects/harvester/data/h/pkls/ADABUSD.pkl")
else:
    print("No new data.")
"""

past_data = norm_augment(past_data)




#outputs the probability for ZERO (red) -> list of 1 !
next_candle_pred = predict(past_data.iloc[-1 : , : ])
print(past_data.index[-1], next_candle_pred[0])
print(f_test_start_date, f"prediction for next candle is {next_candle_pred}")
exit()


"""

#get curent wallet ballance

busd, ada, slac, total, ada_price = get_ada_busd_ballance()

last_close_price = past_data.loc[past_data.index[-1] , "close"]

exposure = busd - next_candle_pred[0]

if abs(exposure) < slac:
    print(past_data.index[-1], "No action")
    # Negative exposure means I'm not holding sufficient busd, positive exposure means too much busd and should buy ada
elif exposure > 0 : 
    print(past_data.index[-1], "Need to buy more ADA")
    am_to_buy = (abs(exposure) * total) / ada_price
    q = round (am_to_buy, 1)
    buy('ADABUSD', q, ada_price)
    print(f'{past_data.index[-1]} bought {q} ADA')
else:
    print(past_data.index[-1], "Overexposed : Need to sell ADA")
    am_to_sell = (abs(exposure) * total) / ada_price
    q = round (am_to_sell, 1)
    sell('ADABUSD', q, ada_price)
    print(f'{past_data.index[-1]} sold {q} ADA')
 

actions = pd.read_csv("./data/actions.csv", names=['date', 'proba_0'])
actions.set_index('date', inplace=True)
actions.index = pd.to_datetime(actions.index)
actions["pred"] = [0 if x > 0.5 else 1 for x in actions.loc[:, "proba_0"]]
reality = past_data.loc[actions.index, "movement"]


y_real = reality.values
y_pred = actions['pred'].values

print (y_real)
print(y_pred)

print('Accuracy:', accuracy_score(y_real, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_real, y_pred))
print('Classification report:\n', classification_report(y_real, y_pred))
"""
test = past_data.loc[:, ["movement","past_2_movement", "past_3_movement", "past_4_movement"]]


def simple_pred (df):
    x = [sum ([df.loc[i , "past_2_movement"], df.loc[i , "past_3_movement"]]) for i in df.index]
    return x

y_real = test["movement"].values
y_pred = [1 if x > 1 else 0 for x in simple_pred(test)]


print('Accuracy:', accuracy_score(y_real, y_pred))
print('Confusion matrix:\n', confusion_matrix(y_real, y_pred))
print('Classification report:\n', classification_report(y_real, y_pred))