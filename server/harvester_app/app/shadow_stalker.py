from binance import ThreadedWebsocketManager
from operations import Ops
from buffer import RollingBufferMixin
import datetime
from time import sleep
from matplotlib.dates import date2num
import pandas as  pd
from collections import deque
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

class PriceTracker(Ops, RollingBufferMixin):
    def __init__(self, pair,di1, df1, pi1, pf1, di2, df2, pi2, pf2, prc_distance = 0.027, momentum = 1, update_frq = 15):
        
        self.Xs = np.array([[datetime.strptime(di1, "%Y-%m-%d %H:%M:%S").timestamp()], 
                            [datetime.strptime(df1, "%Y-%m-%d %H:%M:%S").timestamp()]])  # 2x1 array
        self.ys = np.array([pi1, pf1]).reshape(-1,1)

        # Resistance model data
        self.Xr = np.array([[datetime.strptime(di2, "%Y-%m-%d %H:%M:%S").timestamp()],
                            [datetime.strptime(df2, "%Y-%m-%d %H:%M:%S").timestamp()]])  # 2x1 array
        self.yr = np.array([pi2, pf2]).reshape(-1,1)

        self.support_model = LinearRegression()
        self.resistence_model = LinearRegression()

        self.support_model.fit(X=self.Xs, y=self.ys)
        self.resistence_model.fit(X=self.Xr, y=self.yr)

        self.transations = {"type" : [],
                            "date" : [],
                            "pair" : [],
                            "price": [],
                            "quantity": []
                            }
        super().__init__()
        self._update_frq = update_frq # time interval between price checks
        self.buffer = deque(maxlen=300)
        self._pair = pair
        self._target = 0.3587 #target_price # must start with an a initial price target
        self._prc_distance = prc_distance # a % val that will set the target price
        #self._counter = 0 # number of times the price reaches or passes the target
        self.last_price = 0 
        #self.price_change = 0 # last proice change 
        #self.live_price = {self.pair: None, "error": False} 
        #self.twm = ThreadedWebsocketManager()
        self.order_placed = False  # Added this flag
        self.current_price = None # latest price recived from the socket
        self._momentum = momentum # direction in which the price is being tracked
        self.record_price = None
        self.mock_transations = {"stamp": [],"side": [], "price": [], "amount": []}
        self.mock_usd = 100
        self.mock_mina = 0
        self.up = None
        self._trigger = False
        self._current_timestamp = None
        self._buy_threshold = None
        self._take_profit = None

    
    @property
    def trigger(self):
        return self._trigger
    @trigger.setter
    def trigger(self, val):
        self._trigger = val

    def activate (self):
        self.trigger = True

    def deactivate (self):
        self.trigger = False

    @property
    def current_timestamp(self):
        """Get the current timestamp."""
        return self._current_timestamp

    @current_timestamp.setter
    def current_timestamp(self, value):
        """Set the current timestamp."""
        self._current_timestamp = value

    @property
    def buy_threshold(self):
        """Get the buy threshold value."""
        return self._buy_threshold

    @buy_threshold.setter
    def buy_threshold(self, value):
        """Set the buy threshold value."""
        if not isinstance(value, (int, float)) and value is not None:
            raise ValueError("Buy threshold must be a number or None")
        self._buy_threshold = value

    @property
    def take_profit(self):
        """Get the take profit value."""
        return self._take_profit

    @take_profit.setter
    def take_profit(self, value):
        """Set the take profit value."""
        if not isinstance(value, (int, float)) and value is not None:
            raise ValueError("Take profit must be a number or None")
        self._take_profit = value

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

    def calculate_prices(self, current_timestamp):
        current_timestamp = np.array(current_timestamp).reshape(-1, 1 )
        top_price= self.floor_to_n_digit(self.resistence_model.predict(current_timestamp), 4)
        bottom_price = self.floor_to_n_digit(self.support_model.predict(current_timestamp), 4)
        
        return bottom_price, top_price
    
    def mock_buy(self, q):
        ts = datetime.time()
        self.mock_mina = self.mock_usd / self.target
        self.mock_usd = False
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
    
    def move_target (self):
        if self.up:
            #sets SELL target slightly under the last local high
            self.target =  self.record_price - (self.record_price * self.prc_distance)
        else:
            # Sets BUY target slightly above the last local low
            self.target = self.record_price + (self.record_price * self.prc_distance)

    def activate_alt_btc_tracking(self, msg):
        """Target value must be set prior to calling this method
        Tracks the price evolution realtive to a target price.
        Buys, Sells, Updates target or Pass
        """

        if msg['e'] == '24hrTicker':
            self.current_price = float(msg['c'])
            self.up = self.bought != "BTC" 
            print(f"Current MINA-BTC price --> {self.current_price}")
        
        # pulls the trigger
            c1 = not self.trigger
            c2 = self.up and self.current_price >= self.target # activates when looking to sell
            c3 = not self.up and self.current_price <= self.target # activates wlen looking to buy 

        if c1 and (c2 or c3):
            self.activate()
            self.move_target()
        # starts price trackikg if reigger is activated
        if self.trigger :
            print ("Looking to SELL") if self.up else print("Looking to BUY") 
            #SELL for BTC
            if self.up and self.current_price <= self.target:
                q = self.floor_to_n_digit(self.stash[self.bought], 1)
               
                self.place_spot_order("SELL", self.pair, q, self.target)
                self.target = None
                self.record_price = None
                self.deactivate()
                print(f"Just sold {q}, {self.pair}, at {self.current_price}")
            
            #BUY alt
            elif not self.up and self.current_price >= self.target:
                q = self.floor_to_n_digit(self.stash["BTC"]/self.target, 1)
                self.place_spot_order("BUY", self.pair, q, self.target)
                print(f"Bought {q} MINA")
                self.target= None
                self.record_price = None
                self.deactivate()
            
            #update record price and target
            elif self.up and self.current_price > self.record_price or not self.up and self.current_price < self.record_price:
                self.record_price = self.current_price
                self.move_target()
                print(f"New target is {self.target}")

    def look_to_buy(self, msg): 
        if msg['e'] == '24hrTicker':
            self.current_price = self.floor_to_n_digit(float(msg['c']), 4 )
            if not self.record_price:
                self.record_price = self.current_price
                self.target = self.floor_to_n_digit(self.record_price + (self.record_price * self.prc_distance), 4)
            elif self.current_price > self.target: 
                q = self.floor_to_n_digit(self.stash["USDT"]/self.target, 1)
                price = self.target
                self.place_spot_order("BUY", self.pair, q, price)
                print(f"Bought {q} {self.pair}")
                self.target= None
                self.record_price = None
            elif self.current_price < self.record_price:
                self.record_price = self.current_price

                self.target = self.floor_to_n_digit(self.record_price + (self.record_price * self.prc_distance),4)
                print(f"Target moved to: {self.target}")

    def look_to_sell(self, msg): 
        
        if msg['e'] == '24hrTicker':
            self.current_price = self.floor_to_n_digit(float(msg['c']), 4 )
            if not self.record_price:
                self.record_price = self.current_price
                self.target = self.floor_to_n_digit(self.record_price - (self.record_price * self.prc_distance), 4)
            elif self.current_price < self.target: 
                q = self.floor_to_n_digit(self.stash[self.bought], 1)
                price = self.floor_to_n_digit(self.target, 4)
                self.place_spot_order("SELL", self.pair, q, price)
                self.target = None
                self.record_price = None
                print(f"Just sold {q}, {self.pair}, at {self.current_price}")
            elif self.current_price > self.record_price:
                self.record_price = self.current_price
                self.target = self.floor_to_n_digit(self.record_price + (self.record_price * self.prc_distance),4)
                print(f"Target moved to: {self.target}")


    def handle_ticker(self, msg = None):
        # Callback method Method that handles the tracking and logic. 

        if not msg :
            pass

        if msg['e'] == '24hrTicker':
            self.current_price = self.floor_to_n_digit(float(msg['c']), 4 )
            self.up = self.bought not in ["USDT", "USDC"]

        # Calculate top and bottom prices from grid slopes
            print(f"Current  price {self.current_price} USD")
            self.current_timestamp = datetime.now().timestamp()
            self.buy_threshold, self.take_profit = self.calculate_prices(self.current_timestamp)
            print(f'Activated: {self.trigger}\nactive between {self.buy_threshold} and {self.take_profit}\nGoing UP - {self.up}\n')
            self.check_oo()

            c1 = not self.trigger
            c2 = self.up and self.current_price >= self.take_profit # activates when looking to sell
            c3 = not self.up and self.current_price <= self.buy_threshold # activates wlen looking to buy 
            
            #When the price moves outside the interval start tracking for buy of sell oporunity
            #to be continued
            if c1 and (c2 or c3):
                self.activate()
                self.record_price = self.current_price
                self.move_target()
                print(f"Target designated {self.record_price}")

            #move target
            elif c1:
                pass

            elif (self.up and self.current_price > self.record_price) or not self.up and self.current_price < self.record_price:
                self.record_price = self.current_price
                self.move_target()
                print(f'Updated target to {self.target}')
                        

            elif self.up and self.current_price <= self.target:
                q = self.floor_to_n_digit(self.stash[self.bought], 1)
                price = self.floor_to_n_digit(self.target, 4)
                self.place_spot_order("SELL", self.pair, q, price)
                self.target = None
                self.record_price = None
                print(f"Just sold {q}, {self.pair}, at {self.current_price}")
                self.deactivate()


            elif not self.up and self.current_price >= self.target:
                q = self.floor_to_n_digit(self.stash["USDT"]/self.target, 1)
                price = round(self.target, 4)
                self.place_spot_order("BUY", self.pair, q, price)
                print(f"Bought {q} {self.pair}")
                self.target= None
                self.record_price = None
                self.deactivate()
                
            sleep(1)
        else:
            print(f"Current prce {self.current_price} within range, activation at {self.take_profit}:  Monitoring ...  ")
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

            #Looking to sell
            if self.mock_usd == 0 and self.current_price <= self.target :
                print(f"Looking to sell {self.bought}")
                q = self.floor_to_n_digit(self.mock_mina, self.get_roundings()[1][self.pair])
                self.mock_sell(q)
                self.target = None
                self.record_price = None
                print(f"Just sold {q}, {self.pair}, at {self.current_price}")


            #Looking to BUY 
            elif self.mock_mina == 0 and self.current_price >= self.target:
                q = self.floor_to_n_digit(self.mock_usd/self.target, self.get_roundings()[1][self.pair])
                self.mock_buy(q)
                print(f"Bought {q}, {self.target} for {self.target}")
                self.target = None
                self.record_price = None
            else : print (self.target, self.current_price)

            sleep(1)

    def start_trading(self):
        #self.twm.start()
        #self.twm.start_symbol_ticker_socket(callback=self.handle_ticker, symbol=self.pair)
        print(f"Monitoring {self.pair} target at {self.take_profit} or {self.buy_threshold} up >> {self.up}, acttivated : {self.trigger}")

    def api_calls(self):
        #c = 0
        while True :
            try:
                tick = self.cli.get_symbol_ticker(symbol="ADAUSDT")
                self.current_price = round(float(tick['price']), 4)

            except:
                self.current_price = None

            if not self.current_price:
                while not self.current_price:
                    print("Trying to fetch the price")

                    tick = self.cli.get_symbol_ticker(symbol= self.pair)
                    self.current_price = round(float(tick['price']), 4)
            else : 

            
                self.up = self.bought not in ["USDT", "USDC"]

            # Calculate top and bottom prices from grid slopes
                print(f"Current  price {self.current_price} USD")
                self.current_timestamp = datetime.now().timestamp()
                self.buy_threshold, self.take_profit = self.calculate_prices(self.current_timestamp)
                print(f'Activated: {self.trigger}\nactive between {self.buy_threshold} and {self.take_profit}\nGoing UP - {self.up}\n')
                self.check_oo()

                c1 = not self.trigger
                c2 = self.up and self.current_price >= self.take_profit # activates when looking to sell
                c3 = not self.up and self.current_price <= self.buy_threshold # activates wlen looking to buy 
                
                #When the price moves outside the interval start tracking for buy of sell oporunity
                #to be continued
                if c1 and (c2 or c3):
                    self.activate()
                    self.record_price = self.current_price
                    self.move_target()
                    print(f"Target designated {self.record_price}")

                #move target
                elif c1:
                    pass

                elif (self.up and self.current_price > self.record_price) or not self.up and self.current_price < self.record_price:
                    self.record_price = self.current_price
                    self.move_target()
                    print(f'Updated target to {self.target}')
                            

                elif self.up and self.current_price <= self.target:
                    q = self.floor_to_n_digit(self.stash[self.bought], 1)
                    price = self.floor_to_n_digit(self.target, 4)
                    self.place_spot_order("SELL", self.pair, q, price)
                    self.target = None
                    self.record_price = None
                    print(f"Just sold {q}, {self.pair}, at {self.current_price}")
                    self.deactivate()


                elif not self.up and self.current_price >= self.target:
                    q = self.floor_to_n_digit(self.stash["USDT"]/self.target, 1)
                    price = round(self.target, 4)
                    self.place_spot_order("BUY", self.pair, q, price)
                    print(f"Bought {q} {self.pair}")
                    self.target= None
                    self.record_price = None
                    self.deactivate()
                    
                    sleep(1)
                else:
                    print(f"Current prce {self.current_price} within range, activation at {self.take_profit}:  Monitoring ...  ")
                    sleep(1)
            sleep(7)





    def stop_trading(self):
        self.twm.stop()
        print("Trading stopped.")

def main():
    tracker = PriceTracker(pair="ADAUSDT",
                           di1="2025-03-10 10:00:00",
                           df1="2025-03-16 19:00:00",
                           pi1=0.6607,
                           pf1=0.7495,
                           di2="2025-03-10 10:00:00",
                           df2="2025-03-16 13:00:00",
                           pi2=0.6872,
                           pf2=0.7736,
                           prc_distance= 0.0085)
    #tracker.target = 0.00000361

    #tracker.start_trading()
    tracker.api_calls()

    """try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        pd.DataFrame(tracker.mock_transations).to_pickle("test_data.pkl")
        print("Manually stopped - data saved !")
        print(tracker.mock_usd, tracker.mock_mina)

    finally:
        tracker.stop_trading()"""

if __name__ == "__main__":
    main()