from binance import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
import os
import datetime
import time
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
        self.data_files_pth = "/root/AI_AMA_2/server/harvester_app/data/"
        self.tr_record_pth = "../data/transactions.pkl"
        #self.live_price = {self.pair: None, "error": False}
        self.cli = self.unlock()
        self.pairs = self.get_pairs()
        self.balance = [self.cli.get_asset_balance(c)for c in self._of_interest + ["USDC"]]
        self.stash = dict(zip(self._of_interest + ["USDC"], [float(d["free"]) for d in self.balance if d["asset"] in self._of_interest + ["USDC"]]))
        self.prices_dict = dict(zip(self.pairs, [self.get_price(pair) for pair in self.pairs]))
        self.valuations = self.update_valuations()
        self.bought = self.valuations.idxmax()
        self.usdc = self.bought =="USDC"


    def get_pairs(self):
        pairs = [f'{c}USDC' for c in self._of_interest]
        return pairs

    def update_valuations(self):
        valuations = {}
        for k, val in self.prices_dict.items():
            coin = [c for c in self._of_interest if c in k][0]
            valuations[coin] = 0 if not val else val * self.stash[coin]
        valuations["USDC"] = self.stash["USDC"]
        valuations = pd.Series(valuations)
        return valuations

    def update_transactions_record(self, tr_dict):
        """
        Append a dictionary as a row to a pandas DataFrame stored in a pickle file.
        If the file doesn't exist, create it with the dictionary as the first row.
        Parameters:
        -----------
        data_dict : dict
            Dictionary to append as a row to the DataFrame
        pickle_path : str
            Path to the pickle file
        Returns:
        --------
        None
        """
        # Create a DataFrame from the dictionary (single row)
        tr_df = pd.DataFrame([tr_dict])
        # Check if the pickle file exists
        if os.path.exists(self.tr_record_pth):
            try:
                # Read existing DataFrame from pickle
                existing_df = pd.read_pickle(self.tr_record_pth)
                # Append the new row to the existing DataFrame
                combined_df = pd.concat([existing_df, tr_df], ignore_index=True)
                # Save the combined DataFrame back to the pickle file
                combined_df.to_pickle(self.tr_record_pth)
                print(f"Successfully appended data to existing pickle file: {self.tr_record_pth}")
            except Exception as e:
                print(f"Error appending to pickle file: {e}")
        else:
            try:
                # Create a new pickle file with the DataFrame
                tr_df.to_pickle(self.tr_record_pth)
                print(f"Successfully created new pickle file: {self.tr_record_pth}")
            except Exception as e:
                print(f"Error creating pickle file: {e}")



    def unlock (self, fname = 'nancy.txt'):
        """Returns the Binance Client"""
        with open(fname) as f:
            lines = f.readlines()
        a = lines[0].splitlines()[0]
        b = lines[1]
        return Client(a , b, requests_params={"timeout": 300})

    def get_price(self, pair):
        cli = self.unlock()

        if not pair in self.get_pairs():
            raise ValueError(f"> Available pairs are : \n{self.get_pairs()}\nYou provided {pair}\n")

        # this gets the last traded price (no info on tranzaction type)
        try:
            latest_price = cli.get_symbol_ticker(symbol = pair)
            price = self.floor_to_n_digit( float(latest_price['price']), self.rop[pair])
        except:
            price = None


        return price

    def floor_to_n_digit(self, value, n):
        print(f"Flooring the {value}to {n} digits")
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
        # top_price = df['high'].max()
        # bottom_price = df['low'].min()
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
    def update_pair (self, pair):
        if not os.path.exists(f"{self.data_files_pth}h/{pair}.csv"):
            print(f"Fetching h candles for {pair}")
            self.get_candles(pair)
        else:
            self.update_candles(pair)

    def get_candles(self,pair, start_time, interval = "1h"):
        """Retrives H candle data from binance for a given trading pair 
        parameters:
        - pair: str eg. BTCUSDC
        - start_time int or str timestamp in miliseconds from when to retrive the data uo to present
        - interval: object by default H
        """
        candles = self.cli.get_historical_klines(
        symbol=pair,
        interval = interval,
        start_str=start_time)

        processed_candles = []
        for candle in candles:
            processed_candles.append([
                candle[0],  # open_time
                candle[1],  # open
                candle[2],  # high
                candle[3],  # low
                candle[4],  # close
                candle[5],  # volume
                candle[8]   # number_of_trades
            ])

        # Create DataFrame with only the interesting columns
        df = pd.DataFrame(processed_candles, 
                          columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades'])

        # Convert timestamp to datetime
        df['open_datetime'] = pd.to_datetime(df['open_time'], unit='ms')
        return df


    def update_candles(self, pair, interval = "1h", n = 300):
        """
        Update candle data by:
        1. Reading the last timestamp from the existing CSV file
        2. Fetching new candles from Binance starting after that timestamp
        3. Appending the new data to the CSV file without duplicates
        Parameters:
        - symbol: pair (e.g., "BTCUSDT")
        - interval: default 1h  (e.g "1h")
        - n: Number of hours in the past (300 by default) used only the first time
        """
        stored_data_pth = f'{self.data_files_pth}/h/{pair}.csv'

        if os.path.exists(stored_data_pth):
            with open(stored_data_pth, 'r') as f:
                stored_data = pd.read_csv(f)
                last_timestamp = int(stored_data['open_time'].iloc[-1]) +1

            #update with latest data
            new_df = self.get_candles(pair, start_time = last_timestamp)
            if new_df.empty:
                print(f"No new data to add for {pair}")
            else:
                new_df.to_csv(stored_data_pth, mode='a', header=False, index=False)
                print(f"Added {len(new_df)} new candles to {stored_data_pth}")

        else:
            #get last n candles
            start_time = int(time.time() * 1000) - (n * 60 * 60 * 1000)
            df = self.get_candles(pair, start_time)
            df.to_csv(stored_data_pth, index = False)


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
            if side == "SELL":
                print(f"SOLD! {quantity} of {pair}")
            elif side =="BUY":
                print(f"BOUGHT! {quantity} of {pair}")
            self.valuations = self.update_valuations()

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

if __name__ == "__main__":
    Op = Ops()
    Op.pairs = ["BTCUSDC", "ADAUSDC", "SUIUSDC", "PYTHUSDC", "JUPUSDC"]
    Op.btc_pairs = ["ADABTC", "SUIBTC", "PYTHBTC"]
    for pair in Op.pairs + Op.btc_pairs:
        Op.update_candles(pair)
    print("Candles update successfull !")

