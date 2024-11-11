from binance import ThreadedWebsocketManager
from operations import Ops
from buffer import RollingBufferMixin
import datetime
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException
from matplotlib.dates import date2num
import math
import pandas as  pd
import pickle
from collections import deque

class PriceTracker(Ops, RollingBufferMixin):
    def __init__(self, pair, prc_distance = 0.027, momentum = 1, update_frq = 15):
        super().__init__()
        self._update_frq = update_frq # time interval between price checks
        self.buffer = deque(maxlen=300)
        self._pair = pair
        self._target = 0.56 #target_price # must start with an a initial price target
        self._prc_distance = prc_distance # a % val that will set the target price
        #self._counter = 0 # number of times the price reaches or passes the target
        self.last_price = 0 
        #self.price_change = 0 # last proice change 
        #self.live_price = {self.pair: None, "error": False} 
        self.twm = ThreadedWebsocketManager()
        self.order_placed = False  # Added this flag
        self.current_price = None # latest price recived from the socket
        self._momentum = momentum # direction in which the price is being tracked
        self.record_price = None
        self.mock_transations = {"stamp": [],"side": [], "price": [], "amount": []}
        self.mock_usd = 100
        self.mock_mina = 0
        self.up = None



    @property
    def update_frq (self):
        return self._update_frq
    
    @update_frq.setter
    def update_frq(self, new_frq):
        self._update_frq = new_frq
            
    @property
    def momentum(self):
        return self._momentum
    
    @momentum.setter
    def momentum(self , val):
        self._momentum = val
        
    '''
    @property
    def cli(self):
        return self._cli

    @cli.setter
    def cli(self, client):
        self._cli = client
    '''
    @property
    def pair(self):
        return self._pair

    @pair.setter
    def pair(self, new_pair):
        self._pair = new_pair

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target):
        self._target = new_target

    @property
    def counter(self):
        return self._counter

    @counter.setter
    def counter(self, new_count):
        self._counter = new_count

    @property
    def prc_distance(self):
        return self._prc_distance

    @prc_distance.setter
    def prc_distance(self, new_prc_distance):
        self._prc_distance = new_prc_distance
    
    def mock_buy(self, q):
        ts = datetime.time()
        self.mock_mina = self.mock_usd / self.target
        self.mock_usd = 0
        self.mock_transations["stamp"].append(ts)
        self.mock_transations["side"].append("buy")
        self.mock_transations["price"].append(self.target)
        self.mock_transations["amount"].append(q)

    def mock_sell(self, q):
        
        ts = datetime.time()
        self.mock_usd = self.mock_mina * self.target
        self.mock_mina = 0
        self.mock_transations["stamp"].append(ts)
        self.mock_transations["side"].append("sell")
        self.mock_transations["price"].append(self.target)
        self.mock_transations["amount"].append(q)


    def get_credentials(self, fname='nancy.txt'):
        """Returns the Binance API credentials"""
        with open(fname) as f:
            lines = f.readlines()
        return lines[0].strip(), lines[1].strip()
    
    def check_oo (self):
        while self.cli.get_open_orders(symbol = self.pair ):
            print("Waiting fror order to fill ... ")
            sleep(15)

    def handle_ticker(self, msg):
        # Callback method Method that handles the tracking and logic. 

        if msg['e'] == '24hrTicker':
            self.current_price = self.floor_to_n_digit(float(msg['c']), 4 )
            self.up = self.bought not in ["USDT", "USDC"]


            print(f'Target --> {round(self.target, 4)}\nPRICE --> {self.current_price}\n\n')
            self.check_oo()

            # updating record price : 3 scenarios 
            if not self.record_price or (
                self.up and self.current_price > self.record_price
                ) or (
                    not self.up and self.current_price < self.record_price
                    ):
                self.record_price = self.current_price
                print(f"Updated trigger to {self.record_price}")
            
            self.target = self.record_price - (self.record_price * self.prc_distance) if self.up else self.record_price + (self.record_price * self.prc_distance)
            

            if self.up and self.current_price <= self.target:
                q = self.floor_to_n_digit(self.stash[self.bought], self.get_roundings()[1][self.pair])
                price = self.floor_to_n_digit(self.target, 4)
                self.place_spot_order("SELL", self.pair, q, price)
                self.target = None
                self.record_price = None
                print(f"Just sold {q}, {self.pair}, at {self.current_price}")


            elif not self.up and self.current_price >= self.target:
                q = self.floor_to_n_digit(self.stash["USDT"]/self.target, self.get_roundings[1][self.pair])
                price = round(self.target, 4)
                self.place_spot_order("BUY", self.pair, q, price)
                self.target= None
                self.record_price = None
                print("price is rizing I should buy")
            print (self.record_price)

            sleep(1)

    def fw_test(self, msg):


        if msg['e'] == '24hrTicker':

            self.current_price = self.floor_to_n_digit(float(msg['c']), 4 )
            self.add_value(self.current_price)

            if len(self.buffer) == 300:
                unq_steps = self.get_unique_values()
                step_sizes = self.get_pace_list()
                print(unq_steps)
                print(f"Moves made in, the last 5 min {step_sizes}")
                print(f"Price changed {len(unq_steps)} times in 5 min ")
                print(f'Speed:{len(unq_steps)/5} steps/minute')
                print(f'Average step size: {round(sum(step_sizes)/len(step_sizes), 2 ) } for teh past 5 min' )    


            print(f'current target is {self.target} while the current price is : {self.current_price}')
           
            # updating record price : 3 scenarios 
            if not self.record_price or (
                self.mock_usd == 0 and self.current_price > self.record_price
                ) or (
                    self.mock_mina == 0 and self.current_price < self.record_price
                    ):
                #update price of reference
                self.record_price = self.current_price
                print(f"Updated reference price to {self.record_price}")
            
            self.target = round(self.record_price - (self.record_price * 0.01) if self.mock_usd==0 else self.record_price + (self.record_price * 0.01), 4)
            print (f"New target is {self.target}")

            #Looking toi sell
            if self.mock_usd == 0 and self.current_price <= self.target :
                print(f"Looking to sell {self.bought}")
                q = self.floor_to_n_digit(self.mock_mina, self.get_roundings()[1]["MINAUSDT"])
                self.mock_sell(q)
                self.target = None
                self.record_price = None
                print(f"Just sold {q}, {self.pair}, at {self.current_price}")


            #Looking to BUY 
            elif self.mock_mina == 0 and self.current_price >= self.target:
                q = self.floor_to_n_digit(self.mock_usd/self.target, self.get_roundings()[1]["MINAUSDT"])
                self.mock_buy(q)
                print(f"Bought {q}, {self.target} for {self.target}")
                self.target = None
                self.record_price = None
            else : print (self.target, self.current_price)

            sleep(1)

            """ 


            if (self.current_price <= self.target) and not self.order_placed:
                self.counter +=1  

                if self.counter >= 3:
                    print('I will place a sell order')
                    # Implement your buy order logic here
                    self.order_placed = True
                    self.counter = 0
                    self.momentum = -1
                else:
                    print (f"Counter at {self.counter}, still monitorint ")
                    print(self.current_price)
                    print(self.prc_distance)

            else:
                # This updates teh limit price only if last price is higher than previous price 
                if self.current_price > self.last_price :  
                    self.target = self.current_price - (self.current_price * self.prc_distance * self.update_frq)
                    self.target = self.floor_to_n_digit(self.target, 4)

                    print(f"updatig ... price is {self.current_price} ")
                    print("Holding: ", self.bought)
                    exit()
                sleep(15)
                """
        


    def start_trading(self):
        self.twm.start()
        self.twm.start_symbol_ticker_socket(callback=self.handle_ticker, symbol=self.pair)
        print(f"Monitoring {self.pair} price. Target: {self.target}")


    def stop_trading(self):
        self.twm.stop()
        print("Trading stopped.")

def main():
    tracker = PriceTracker(pair="MINAUSDT", prc_distance= 0.0085)
    tracker.start_trading()

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        pd.DataFrame(tracker.mock_transations).to_pickle("test_data.pkl")
        print("Manually stopped - data saved !")
        print(tracker.mock_usd, tracker.mock_mina)

    finally:
        tracker.stop_trading()

if __name__ == "__main__":
    main()