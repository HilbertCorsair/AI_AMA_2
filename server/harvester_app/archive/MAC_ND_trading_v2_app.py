#!/usr/bin/env python3
from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
from time import sleep
import time
import pickle
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import xgboost as xgb
import pickle
import copy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<| GO! |>~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]

mins = [
    15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
    869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230, 1.1
    ]
tops = [
    69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
    4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85, 19.23
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



#modeled = ["BTC", "MINA",  "AGIX", "ALGO", "MATIC", "KAVA", "FIL", "RNDR", "ADA", "BNB", "AR"]
#data_files =  [f'/home/honeybadger/projects/harvester/data/h/{c}BUSD.csv'for c in of_interest]

def ma_fnc():
    return

def unlock (fname = '/home/honeybadger/projects/harvester/nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

# imports data from csv files
def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

# Updating the Coins data dictionary
def update_pair (pair,timestamp):
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

def get_price(pair):
    cli = unlock()
   # this gets the last traded price (no info on tranzaction type)
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, rounding_order_price[pair])
    return price

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
        print(f" Placing SELL {pair} order ...")
        #open_order = cli.get_open_orders(symbol = pair )
        while cli.get_open_orders(symbol = pair ):
            #orderID = open_order[0]['orderId']
            if c1 < 1 :
                print("waiting for order to be filled ...")
                c1 += 1
        
            else : 
                c1 += 1
                sleep(30)

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
                sleep(30)
        #print("DONE!")

    except BinanceAPIException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)

def predict (series, pair):
    #add proba
    print("making predition now")
    with open(f'/home/honeybadger/projects/harvester/models/{pair}_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    # Use the loaded model to make predictions
    prediction =  model.predict_proba(series)
    print(f'I predict {prediction}')
    
    return prediction[0]

def ballancer (rec_exposure, lbl ):
    
    cli = unlock()
    print("Checking evaluations ... ")
    btc_price = get_price(f'BTCBUSD')
    
    balance = [cli.get_asset_balance(c)for c in ["BUSD", "BTC"] ]
    stash = dict(zip(["BUSD", "BTC"], [float(d["free"]) for d in balance]))
    # Determining the holding position
    stash_eval = stash["BUSD"] + stash["BTC"] * btc_price
    btc_exposure = (stash["BTC"] * btc_price) / stash_eval
    print(f"BTC exposure is {round(btc_exposure , 2)} while recomandat exposure is {round(rec_exposure , 2)}")

    if abs(rec_exposure - btc_exposure) < 0.1 : #if within 10%
        print(f"{lbl} No need to act")
    else:
        if rec_exposure > btc_exposure:

            print(f"Need to buy {(rec_exposure-btc_exposure)*stash_eval} worth of BTC")
            q = round (
                    ((rec_exposure-btc_exposure)*stash_eval) / btc_price
                            , rounding_order_crypro_amount["BTCBUSD"]
                            )
            buy('BTCBUSD', q, btc_price)
            print(f'bought{q} BTC {lbl}')

        else:
            print(f"Need to sell {(btc_exposure - rec_exposure )*stash_eval} worth of BTC")
            q = round (
                    ((btc_exposure - rec_exposure )*stash_eval) / btc_price
                            ,rounding_order_crypro_amount["BTCBUSD"]
                            )
            sell('BTCBUSD', q, btc_price)
            print(f'sold{q} BTC {lbl}')

#=======================================================================================================================
# TS for last 100 candels
#c_data = [import_coin_data(pth) for pth in data_files] #from old csvs !!! ALSO ADDS DATES AS INDEXES 
c_data = [pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl") for p in pairs]
c_dict = dict(zip(pairs, c_data))

#get latest timestamp
# timestamp() gets ts from date-time index in secounds. 
# timestamp needs to be in miliseconds and as a string to be used in the update_pair fuction
# IS UPDATED BY updating c_dict and pickle data
# !!! When timestamp in the request is string it is excluded from the recived data. If up to date the respons is an empty array 
#file data check
# # UPDATE FILES AND REWRITE DATA 

for p in pairs:
    
    timestamp =  str(int(c_dict[p].index[-2].timestamp())*1000)
    # This part is useless if data in c_dict is up to date
    new_df = update_pair(p)
    if not new_df.empty:
        # Add data to existing dict and also save updatated to pkl
        c_dict[p] = pd.concat([c_dict[p], new_df], axis=0)
        c_dict[p].to_pickle(path = f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl")
        print(f"{p} updated")
    else:
        print("No new data.")

# Separating the data saved from the data to be processed since the processing would afect
#   the original c_dict data that then gets overwritten and saved to files !!!
#   The new op_dict makes sure processed data is not stored to hystorical data files

for pair, df in c_dict.items():
    c_dict[pair] = norm_augment(df).tail(159) #keep only the last 159 hours
    print("Data augmented")
    cols = c_dict[pair].columns   
    c_dict[pair].columns = [f'{pair}_{col}' for col in cols ]
#print(op_dict)
print ('exiting loop')
lbl = op_dict['BTCBUSD'].index[-1]

#----------------------------------------------------
print(lbl)
pred = round(predict(op_dict["BTCBUSD"].iloc[-1, : ], "BTCBUSD") , 2)
#print(op_dict["BTCBUSD"].index[-1:])
print(f'Made the prediction {pred} for {lbl}')

#pred_df.sort_values(ascending=False, inplace=True)

# Change the column names of the dataframes to include the pair so thei can be joined

        
#last_candle = pd.concat([df.iloc[-1, : ] for df in op_dict.values()], axis=0)


print("Appling strategy...")
t_zero = time.time()
ballancer(rec_exposure = pred, lbl = lbl)
t1 = time.time()
job_time = int(t1 - t_zero)

print(f"It took {job_time} seconds !\n")

