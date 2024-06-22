from binance import ThreadedWebsocketManager
from operations import Ops
import datetime
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import math
import pandas as pd
import numpy as np

class Sdt48 (Ops):
    def __init__(self, pair, line_dict = False):
        super().__init__()
        self.switch = False # False for not holding crypto (waiting to buy)
        self.problem = False
        self.transations = {"type" : [],
                            "date" : [],
                            "pair" : [],
                            "price": [],
                            "quantity": []
                            }

        self.cli = self.unlock()
        self._pair = pair
        self._line_dict = line_dict
        self._candels = self.get_candels(48)
        self.mean = self.get_mean_sd()[0]
        self.sd = self.get_mean_sd[1]

        self.live_price = {self.pair : None, "error": False}
        
    
    @property
    def pair(self):
        return self._pair
    @pair.setter
    def pair(self, val):
        self._pair = val

    
    @property
    def line_dict(self):
        return self._line_dict
    
    @line_dict.setter
    def line_dict(self, dict):
        self._line_dict = dict
    
    @property
    def candels(self):
        return self._candels
    @candels.setter
    def candels(self, df):
        self._candels = df

    def flip (self, bol):
         return not bol
    
    #def record_transaction():
    def get_candels(self, number = 48):
        df = self.cli.get_klines(symbol= self.pair, interval=self.cli.KLINE_INTERVAL_1HOUR, limit=number)
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 
           'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']

        df = pd.DataFrame(df, columns=columns)

        # Convert the timestamp columns to datetime
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

        df["High"] = pd.to_numeric(df['High'])
        df["Low"] = pd.to_numeric(df['Low'])

        return df.loc[:["High", "Low"]]

    def get_mean_sd(self):
        # Display the DataFrame
        values = list(self.candels['High'])
        values += list(self.candels['Low'])

        mean_sd = [np.mean(values), np.std(values, ddof=1)]
        mean_sd= [self.floor_to_n_digit(x , 4) for x in mean_sd]

        return mean_sd
       
    def get_line_slope(self):

            time_interval = self.line_dict['date_f'].timestamp() - self.line_dict['date_i'].timestamp()
            price_interval = self.line_dict['price_f'] - self.line_dict["price_i"]
            slope = price_interval / (time_interval) #dollars per h


            return slope


    def live_trade_pair(self):

            s = self.get_line_slope() if self.line_dict else 1

            bsm = ThreadedWebsocketManager()
            bsm.start()
            bsm.start_symbol_ticker_socket(symbol=self.pair, callback= self.pairs_trade)
            while not self.live_price[self.pair]:
                sleep(0.1)

            while True:
                
                if self.live_price['error']:
                    bsm.stop()
                    sleep(3)
                    bsm = ThreadedWebsocketManager()
                    bsm.start()
                    bsm.start_symbol_ticker_socket(symbol=self.pair, callback= self.pairs_trade)
                    self.live_price["error"] = False

                else:
                    print(self.live_price[self.pair])
                    bottom = (self.mean - self.sd) * s
                    bottom = round(bottom, 4)
                    top  = (self.mean + self.sd) *s
                    top = round(top, 4)

                    print(f"\nbottom price : {bottom}\ntop price: {top}\n")

                    if self.switch:
                                             
                        if self.live_price[self.pair] <= bottom :#or self.live_price[pair] >= take_profit
                            try :
                                balance = self.cli.get_asset_balance(asset='USDT')
                                balance = float(balance["free"])
                                if balance < 10:
                                    print("no cash !")
                                    bsm.stop()
                                    
                                    self.problem = self.flip(self.problem)
                                    break

                                q = math.floor(balance/self.live_price[self.pair]) - 5
                                self.place_spot_order("BUY", self.pair, q , self.live_price[self.pair])


                                self.switch = self.flip(self.switch)
                                print("Just bought")

                            except Exception as e:
                                print(e)

                                self.problem = self.flip(self.problem)
                                break
                        else : 
                            print("Price within range: waiting for buy signal ... ")
                    else:
                         
                        if self.live_price[self.pair] < top :
                            rounding_order_price, rounding_order_crypro_amount = self.get_roundings()
                            try :
                                balance = self.cli.get_asset_balance(asset='MINA')
                                balance = float(balance["free"])
                                q = self.floor_to_n_digit(balance, rounding_order_crypro_amount[self.pair])
                                if self.live_price[self.pair] * q <= 10:
                                    bsm.stop()
                                    print("No asset!")
                                    self.problem = self.flip(self.problem)
                                    break

                                p = self.floor_to_n_digit(self.live_price[self.pair], rounding_order_price[self.pair]) 

                                self.place_spot_order("SELL", self.pair, q , p)
                                self.switch = self.flip(self.switch)
                                print("Just sold")

                            except Exception as e:
                                print(e)

                                break
                        else : 
                            print("Price within range: monitoring ... ")
        
                sleep(0.5) if not self.problem else exit()
