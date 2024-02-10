from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import pickle
import statsmodels.api as sm
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

of_interest =   ['BTC', 'ADA', 'MINA', "PAXG", "AGIX", "DOT", "AR", "LINK"]
pairs = [f'{c}USDT' for c in of_interest]


BUSD_decs = [2, 4, 3, 0, 5, 2, 3, 2]
C_decs = [5, 1, 1, 4, 2, 3, 2, 3]

rounding_order_price = dict(zip(pairs, BUSD_decs))
rounding_order_crypro_amount = dict(zip(pairs, C_decs))

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)
cli = unlock()
def get_price(pair):
    if not pair in pairs:
        raise ValueError(f"> Available pairs are : \n{pairs}\nYou provided {pair}\n")
   # this gets the last traded price (no info on tranzaction type)
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, rounding_order_price[pair])
    return price



def import_coin_data(p, from_pkl = True):
    """ import coin pair data from pkl files by default or from csv files
     - p is a coin pair string exemple : BTCUSDT
     - returns a pandas dataframe with date-time index
    """
    if from_pkl:
        df = pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl")
    else:
        df = pd.read_csv(f"/home/honeybadger/projects/harvester/data/h/{p}.csv",
                         names=['date', 'open', 'high', 'low', 'close'])
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')

    return df


def norm_augment(df) :
    rng = range(len(df.index))
    top_price = max(df.loc[:,'high'])
    botom_price = min(df.loc[:,"low"])
    #first : store the average price in the avg col 
    df.loc[:,'avg'] = [(df["high"][i] + df["low"][i])/2 for i in rng]
    #df.loc[:, "prc_spread"] = [[((df["high"][i] - df["low"][i])/(2*df["avg"][i]))*100 for i in rng]]
    # !!! SUPER IMPORTANT price values are overwritten with their normalized values (min - max normalisation)
    # this step makes it possible to compare the MACD values across all coins 
    # MACD values are not calculated on the absolute prices but on their respective normalized values
    df.loc[:,"open"] = [(df["open"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"high"] = [(df["high"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"low"] = [(df["low"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"close"] = [(df["close"][i] - botom_price)/(top_price - botom_price) for i in rng]
    #df.loc[:, "3ema"] = [bl.ema(df['close'][i], period=3)for i in rng]
    #df.loc[:, "7sma"] = bl.sma(df['close'], period=7)

    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    df = df.join(macd.df)
    df = df.tail(1000)
    x = np.array([date2num(d) for d in  df.index])
    y = df.loc[:,'histogram'].values
    lowess = sm.nonparametric.lowess
    df['LOWESS'] = lowess(df.loc[:,'histogram'], x, frac=0.0197)[:, 1]
    df.loc[:, "momentum"] = np.gradient(x, y)
    return 


# Updating the Coins data dictionary
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

def sell( pair, q , price):
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

#==========================================================================================================
