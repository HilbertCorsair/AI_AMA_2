from binance import ThreadedWebsocketManager
from operations import Ops
import datetime
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import math

class TradeBot (Ops):
    def __init__(self):
        super().__init__()
        self.switch = False # False for not holding crypto (waiting to buy)
        self.problem = False
        self.transations = {"type" : [],
                            "date" : [],
                            "pair" : [],
                            "price": [],
                            "quantity": []
                            }

        self.live_price = {"MINAUSDT": None, "error": False}
        self.cli = self.unlock()

        self.grid_params = {
            "support_date_i" : datetime.datetime.strptime("2024-05-15 02:00:00", "%Y-%m-%d %H:%M:%S"),
            "support_date_f": datetime.datetime.strptime("2024-06-12 10:00:00", "%Y-%m-%d %H:%M:%S"),
            "support_price_i": 0.7254,
            "support_price_f": 0.8558,

            "top_date_i":datetime.datetime.strptime("2024-05-26 08:00:00" ,"%Y-%m-%d %H:%M:%S"),
            "top_date_f": datetime.datetime.strptime("2024-06-13 06:00:00", "%Y-%m-%d %H:%M:%S"),
            "top_price_i": 0.8662,
            "top_price_f": 0.8833
            }
    def flip (self, bol):
         return not bol
    
    #def record_transaction():
         
       
    def get_grid_slopes(self):

            support_time_interval = self.grid_params['support_date_f'].timestamp() - self.grid_params['support_date_i'].timestamp()
            support_price_interval = self.grid_params['support_price_f'] - self.grid_params["support_price_i"]
            support_slope = support_price_interval / (support_time_interval) #dollars per h


            top_time_interval = self.grid_params['top_date_f'].timestamp() - self.grid_params['top_date_i'].timestamp()
            top_price_interval = self.grid_params['top_price_f'] - self.grid_params["top_price_i"]
            top_slope = top_price_interval / (top_time_interval) #dollars per h

            return support_slope, top_slope


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

                    if self.switch:
                                             
                        if self.live_price[pair] <= stop_loss :#or self.live_price[pair] >= take_profit:
                            try :
                                balance = self.cli.get_asset_balance(asset='USDT')
                                balance = float(balance["free"])
                                if balance < 10:
                                    print("no cash !")
                                    bsm.stop()
                                    
                                    self.problem = self.flip(self.problem)
                                    break

                                q = math.floor(balance/self.live_price[pair]) - 5
                                self.place_spot_order("BUY", pair, q , self.live_price[pair])


                                self.switch = self.flip(self.switch)
                                print("Just bought")

                            except Exception as e:
                                print(e)

                                self.problem = self.flip(self.problem)
                                break
                        else : 
                            print("Price within range: waiting for buy signal ... ")
                    else:
                         
                        if self.live_price[pair] < stop_loss :
                            rounding_order_price, rounding_order_crypro_amount = self.get_roundings()
                            try :
                                balance = self.cli.get_asset_balance(asset='MINA')
                                balance = float(balance["free"])
                                q = self.floor_to_n_digit(balance, rounding_order_crypro_amount[pair])
                                if self.live_price[pair] * q <= 10:
                                    bsm.stop()
                                    print("No asset!")
                                    self.problem = self.flip(self.problem)
                                    break

                                p = self.floor_to_n_digit(self.live_price[pair], rounding_order_price[pair]) 

                                self.place_spot_order("SELL", pair, q , p)
                                self.switch = self.flip(self.switch)
                                print("Just sold")

                            except Exception as e:
                                print(e)

                                break
                        else : 
                            print("Price within range: monitoring ... ")
        
                sleep(0.5) if not self.problem else exit()
