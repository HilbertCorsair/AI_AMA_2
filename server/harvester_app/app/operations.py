from binance import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
import os
import datetime
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import math

#import statsmodels.api as sm
from coins import Coins as C 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<| GO! |>~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Ops (C):
    def __init__(self):
        super().__init__()
        # rounding order price and Crypto Amount
        self.rop , self.rca = self.get_roundings()
        self.pkl_files = "/home/honeybadger/projects/harvester/data/h/pkls"
        self.csv_files = "/home/honeybadger/projects/harvester/data/h"
        self.live_price = {"MINAUSDT": None, "error": False}
        self.cli = self.unlock()

        self.grid_params = {
            "support_date_i" : datetime.datetime.strptime("2024-05-14 14:00:00", "%Y-%m-%d %H:%M:%S"),
            "support_date_f": datetime.datetime.strptime("2024-06-02 10:00:00", "%Y-%m-%d %H:%M:%S"),
            "support_price_i": 0.7219,
            "support_price_f": 0.8243,

            "top_date_i":datetime.datetime.strptime("2024-05-09 22:00:00" ,"%Y-%m-%d %H:%M:%S"),
            "top_date_f": datetime.datetime.strptime("2024-05-31 20:00:00", "%Y-%m-%d %H:%M:%S"),
            "top_price_i": 0.7769,
            "top_price_f": 0.8974
            }


    def unlock (self, fname = 'nancy.txt'):
        """Returns the Binance Client"""
        with open(fname) as f:
            lines = f.readlines()
        a = lines[0].splitlines()[0]
        b = lines[1]
        return Client(a , b)
    
    def get_price(self, pair):
        cli = self.unlock()

        if not pair in self.get_pairs():
            raise ValueError(f"> Available pairs are : \n{self.get_pairs()}\nYou provided {pair}\n")
        
        # this gets the last traded price (no info on tranzaction type)
        latest_price = cli.get_symbol_ticker(symbol = pair)
        price = float(latest_price['price'])

        if pair in self.get_pairs():
            price = round(price, self.rop[pair])

        return price

    def floor_to_n_digit(self, value, n):
        scaled_value = value * 10**n
        floored_value = math.floor(scaled_value)
        return floored_value / 10**n
    
    def import_coin_data(self, p, from_pkl = True):
        """ import coin pair data from pkl files by default or from csv files
        - p is a coin pair string exemple : BTCUSDT
        - returns a pandas dataframe with date-time index
        """
        if from_pkl:
            df = pd.read_pickle(f"{self.pkl_files}/{p}.pkl")
        else:
            df = pd.read_csv(f"{self.csv_files}/{p}.csv",
                            names=['date', 'open', 'high', 'low', 'close'])
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index, unit='ms')

        return df

    def norm_augment(self, df): 
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
    
    def update_h_candles(self, pairs):
        try: 
            c_data = [pd.read_pickle(f"{self.pkl_files}/{p}.pkl").tail(468) for p in pairs]
        except:
            data_files =  [f'{self.csv_files}/{pair}.csv'for pair in pairs ] 
            c_data = [self.import_coin_data(pth) for pth in data_files] #from old csvs !!! ALSO ADDS DATES AS INDEXES

        c_dict = dict(zip(pairs, c_data))
        
        for p in pairs:
            timestamp =  int(c_dict[p].index[-1].timestamp())*1000
            # This part is useless if data in c_dict is up to date
            new_df = self.update_pair(p, timestamp).iloc[1 : ]
            
            if not new_df.empty:
                # Add data to existing dict and also save updatated to pkl
                c_dict[p] = pd.concat([c_dict[p], new_df], axis=0)
                c_dict[p].to_pickle(path = f"{self.pkl_files}/{p}.pkl")
                print(f"{p} updated")
            else:
                print("No new data.")
        print("Update complete!")

    def record_transaction(self, df_row):
        file_path = "../data/trecord.csv"
        if not os.path.isfile(file_path):
            df_row.to_csv(file_path, mode = "w", header = True, incex = True)
        else:
            df_row.to_csv(file_path, mode = "a", header = True, incex = True)

    def place_spot_order (self, side, pair, quantity, price):
        cli = self.unlock()
        order_params = {'symbol': pair,
                        'side': side,
                        'type': "LIMIT",
                        'timeInForce': "GTC",
                        'quantity': quantity,
                        'price': price
                        }
        try: 
            cli.create_order(**order_params)
            c1 = 0
            # While loop waits 50 min for order to fill than cancels 
            print(f" Placing SELL {pair} order ...")
            #open_order = cli.get_open_orders(symbol = pair )
            while cli.get_open_orders(symbol = pair ):
                open_order = cli.get_open_orders(symbol = pair)
                orderID = open_order[0]['orderId']
                if c1 >= 40:
                    cli.cancel_order(symbol = pair, orderId = orderID)
                    cli.order_market_buy( symbol=pair,quantity=quantity)

                #orderID = open_order[0]['orderId']
                if c1 < 1 :
                    print("waiting for order to be filled ...")
                    c1 += 1
                else :
                    c1 += 1
                    sleep(30)
            #del(order_params['timeInForce'])
            #order_params['date'] = datetime.datetime.now()
            #order_params = {key:[val] for key, val in order_params.items()}
            #order_params = pd.DataFrame(order_params).set_index("date", inplace = True)
            #self.record_transaction(order_params)
            #print("DONE!")
                    
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
    
    def get_grid_slopes(self):

        support_time_interval = self.grid_params['support_date_f'].timestamp() - self.grid_params['support_date_i'].timestamp()
        support_price_interval = self.grid_params['support_price_f'] - self.grid_params["support_price_i"]
        support_slope = support_price_interval / (support_time_interval) #dollars per h


        top_time_interval = self.grid_params['top_date_f'].timestamp() - self.grid_params['top_date_i'].timestamp()
        top_price_interval = self.grid_params['top_price_f'] - self.grid_params["top_price_i"]
        top_slope = top_price_interval / (top_time_interval) #dollars per h

        return support_slope, top_slope
    
        
    def pairs_trade(self, msg):
        pair = 'MINAUSDT'  
        try:
            if msg['e'] == 'error':
                self.live_price['error'] = True
            else:
                self.live_price[pair] = float(msg['c'])  # Assuming 'c' is the current price field
        except Exception as e:
            print(f"Error in pairs_trade: {e}")
            self.live_price['error'] = True


    def live_trade_pair(self, pair="MINAUSDT"):

        bs , ts = self.get_grid_slopes()
        

        bsm = ThreadedWebsocketManager()
        bsm.start()
        bsm.start_symbol_ticker_socket(symbol=pair, callback= self.pairs_trade)
        while not self.live_price[pair]:
            sleep(0.1)

        while True:
            
            if self.live_price['error']:
                bsm.stop()
                sleep(3)
                bsm = ThreadedWebsocketManager()
                bsm.start()
                bsm.start_symbol_ticker_socket(symbol=pair, callback= self.pairs_trade)
                self.live_price["error"] = False

            else:
                print(self.live_price[pair])
                current_timestamp = datetime.datetime.now().timestamp()
                stop_loss = (current_timestamp - self.grid_params['support_date_i'].timestamp()) * bs + self.grid_params["support_price_i"]
                stop_loss = round(stop_loss, 4)
                take_profit = (current_timestamp - self.grid_params['top_date_i'].timestamp())*ts  + self.grid_params["top_price_i"]
                take_profit = round(take_profit, 4)
                print(f"\nbottom price : {stop_loss}\ntop price: {take_profit}\n")
                """
                if self.live_price[pair] < stop_loss or self.live_price[pair] >= take_profit:
                    rounding_order_price, rounding_order_crypro_amount = self.get_roundings()
                    try :
                        balance = self.cli.get_asset_balance(asset='MINA')
                        balance = float(balance["free"])
                        q = self.floor_to_n_digit(balance, rounding_order_crypro_amount[pair])
                        p = self.floor_to_n_digit(self.live_price[pair], rounding_order_price[pair]) 

                        self.place_spot_order("SELL", pair, q , p)

                        break
                    except Exception as e:
                        print(e)
                        break
                else : 
                    print("Price within range: monitoring ... ")
                """

                if self.live_price[pair] <= stop_loss :#or self.live_price[pair] >= take_profit:
                    try :
                        balance = self.cli.get_asset_balance(asset='USDT')
                        balance = float(balance["free"])
                        if balance < 10:
                            print("no cash !")
                            exit()

                        q = math.floor(balance/self.live_price[pair]) - 5
                        self.place_spot_order("BUY", pair, q , self.live_price[pair])

                        break
                    except Exception as e:
                        print(e)
                        break
                else : 
                    print("Price within range: monitoring ... ")
    
            sleep(0.5)

    #==========================================================================================================

