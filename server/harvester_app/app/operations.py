from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import pickle
import statsmodels.api as sm
from coins import Coins as C 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<| GO! |>~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Ops (C):
    def __init__(self):
        super().__init__()
        # rounding order price and Crypto Amount
        self.rop , self.rca = self.get_roundings()


    def unlock (self, fname = 'nancy.txt'):
        with open(fname) as f:
            lines = f.readlines()
        a = lines[0].splitlines()[0]
        b = lines[1]
        return Client(a , b)
    
    def get_price(self, pair):
        cli = self.unlock()

        #if not pair in self.get_pairs():
        #    raise ValueError(f"> Available pairs are : \n{self.get_pairs()}\nYou provided {pair}\n")
    # this gets the last traded price (no info on tranzaction type)
        latest_price = cli.get_symbol_ticker(symbol = pair)
        price = float(latest_price['price'])
        if pair in self.get_pairs():
            price = round(price, self.rop[pair])
        return price

    def import_coin_data(self, p, from_pkl = True):
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

    def norm_augment(df): 
        #df = df.drop_duplicates().copy()
        # Assuming df is a pandas DataFrame with 'high', 'low', 'open', 'close' columns.
        top_price = df['high'].max()
        bottom_price = df['low'].min()

        # Vectorized operations for efficiency
        df['avg'] = (df['high'] + df['low']) / 2
        df['open'] = (df['open'] - bottom_price) / (top_price - bottom_price)
        df['high'] = (df['high'] - bottom_price) / (top_price - bottom_price)
        df['low'] = (df['low'] - bottom_price) / (top_price - bottom_price)
        df['close'] = (df['close'] - bottom_price) / (top_price - bottom_price)
        df = df.loc[df.index.duplicated(keep='first')]
        macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        df.dropna(inplace=True)

        
        # Convert index to matplotlib date numbers if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            x = np.array([date2num(d) for d in df.index])
        else:
            x = np.arange(len(df))

        y = df['MACDh_12_26_9'].values
        # Momentum calculation
        df['momentum'] = np.gradient(y, x)

        # LOWESS smoothing
        #lowess = sm.nonparametric.lowess(y, x, frac=0.019)
        #df['LOWESS'] = lowess[:, 1]


        return df
    
    # Updating the Coins data dictionary
    def update_pair (self, pair, timestamp):
        cli = self.unlock()
        bars = cli.get_historical_klines(pair, "1h", timestamp)
        #create empty dataframe
        df = pd.DataFrame( columns= ['date', 'open', 'high', 'low', 'close'])
        for bar in bars:
            # add each row to the dataframe
            df.loc[len(df.index)] = [float(x) for x in bar[0:5]]
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')
        return df

    def place_spot_order (self, side, pair, quantity, price):
        cli = self.unlock()
        try: 
            cli.create_order(
                symbol = pair,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity = quantity,
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
            if side == "SELL":
                print(f"SOLD! {quantity} of {pair}")
            elif side =="BUY":
                print(f"BOUGHT! {quantity} of {pair}")

        except BinanceAPIException as e:
            print(f'Oder failed to pass for q {quantity}, with price {price}, for {pair}')
            print(e)
            
        except BinanceOrderException as e:
            # error handling goes here
            print(f'Oder failed to pass for q {quantity}, with price {price}, for {pair}')
            print(e)

    #==========================================================================================================
