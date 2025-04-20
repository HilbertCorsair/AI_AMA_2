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
#import logging ----> add appropriate logging


class PriceTracker(Ops, RollingBufferMixin):
    def __init__(self, pair,di1, df1, pi1, pf1, di2, df2, pi2, pf2, prc_distance = 0.027, momentum = 1, update_frq = 15):

        # Support model data
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

        self.transactions = {"type" : [],
                            "date" : [],
                            "pair" : [],
                            "price": [],
                            "quantity": []
                            }
        super().__init__()
        self._update_frq = update_frq # time interval between price checks
        self.buffer = deque(maxlen=300)
        self._pair = pair
        self._target = 0.6597 #target_price # must start with an a initial price target
        self._prc_distance = prc_distance # a % val that will set the target price
        #self._counter = 0 # number of times the price reaches or passes the target
        self.last_price = 0 
        #self.price_change = 0 # last proice change 
        self.live_price = {self.pair: None, "error": False} 
        self.twm = ThreadedWebsocketManager()
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
        self.cooldown = (False, 0)
        self.freestyle = False


    @property
    def trigger(self):
        return self._trigger
    @trigger.setter
    def trigger(self, val):
        self._trigger = val

    def activate (self):
        self.record_price = self.current_price
        self.move_target()
        self.trigger = True


    def deactivate (self):
        self.trigger = False
        self.record_price = None
        self.target = None

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

    def update_record(self, tr_type, price, quantity):
        self.transactions["date"].append(datetime.now().timestamp())
        self.transactions["type"].append(tr_type)
        self.transactions["pair"].append(pair)
        self.transactions["price"].append(price)
        self.transactions["quantity"].append(quantity)


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
                self.cooldown = (True, -1)

            #BUY alt
            elif not self.up and self.current_price >= self.target:
                q = self.floor_to_n_digit(self.stash["BTC"]/self.target, 1)
                self.place_spot_order("BUY", self.pair, q, self.target)
                print(f"Bought {q} MINA")
                self.target= None
                self.record_price = None
                self.deactivate()
                self.cooldown = (True,1)

            #update record price and target
            elif self.up and self.current_price > self.record_price or not self.up and self.current_price < self.record_price:
                self.record_price = self.current_price
                self.move_target()
                print(f"New target is {self.target}")

    def look_to_buy(self, price): 
        if not self.trigger :
            print(f"Price at {self.current_price}, looking to activate trackin for BUY order at {price}")

            if self.current_price <= price:
                print(f"Activating price tracking for sell order")
                self.activate()
        #Move target
        elif not self.up and self.current_price < self.record_price:
            self.record_price = self.current_price
            self.move_target()
            print(f'Updated target to {self.target}')
            sleep(2)

        #BUY
        elif not self.up and self.current_price >= self.target:
            q = self.floor_to_n_digit(self.stash["USDC"]/self.target, 1)
            price = round(self.target, 4)
            self.place_spot_order("BUY", self.pair, q, price)
            print(f"Bought {q} {self.pair}")
            self.cooldown = (True, 1)
            self.deactivate()
            self.freestyle= True
            sleep(3)
        else:
            print(f"Tracking ... ")

    def look_to_sell(self, price):
        if not self.trigger :
            print(f"Price at {self.current_price}, looking to activate trackin for sell order at {price}")

            if self.current_price >= price:
                print(f"Activating price tracking for sell order")
                self.activate()
        #Move target
        elif (self.up and self.current_price > self.record_price) or not self.up and self.current_price < self.record_price:
            self.record_price = self.current_price
            self.move_target()
            print(f'Updated target to {self.target}')
            sleep(2)

        #SELL
        elif self.up and self.current_price <= self.target:
            q = self.floor_to_n_digit(self.stash[self.bought], 1)
            price = self.floor_to_n_digit(self.target, 4)
            self.place_spot_order("SELL", self.pair, q, price)
            print(f"Just sold {q}, {self.pair}, at {self.current_price}")
            self.cooldown = (True, -1)
            self.deactivate()
            self.freestyle = True
            sleep(2)
        else:
            print(f"Tracking ... ")

    def interval_trading(self):
        print(f"Current  price {self.current_price} USD")
        self.current_timestamp = datetime.now().timestamp()
        
        # Calculate top and bottom prices from linear regression models
        # For constant support and resistence levels just use a touple 
        self.buy_threshold, self.take_profit = self.calculate_prices(self.current_timestamp) #(0.6211, 0.6485)

        print(f'Activate at : {self.trigger}\nactive between {self.buy_threshold} and {self.take_profit}\nGoing UP - {self.up}\n')
        c0 = self.cooldown[0]
        c05 = self.cooldown[1] > 0
        c1 = not self.trigger
        c2 = self.up and self.current_price >= self.take_profit # activates looking to sell
        c3 = not self.up and self.current_price <= self.buy_threshold # activates looking to buy 
        self.check_oo()
        print(f"Conditionals state: {[c0, c05, c1, c2, c3]}")

        #If on cooldown check temperature ---> check if price moves back into the intervall and mind the direction!
        #This is to prevent buying imediatly after selling when outside the intervall.
        if c0 :
            print("On cooldown")
            if c05 : # previously bought
                if c2:
                    self.cooldown = (False, 0)
                    self.activate() #trigger in place ---> looking to sell
                    print("Activated: Looking to sell")
            else:
                if c3 :
                    self.cooldown = (False, 0)
                    self.activate() #trigger in place ---> looking to buy
                    print("Activated: Looking to buy")

        elif c1 and (c2 or c3):
            self.activate()
            print(f"Target designated {self.record_price}")

        elif c1 :
            print("Price within range. Monioring ...")
            if self.bought:
                print(f"{self.current_price} --Activate LTS @--> {self.take_profit}")

            else:
                print(f"{self.current_price} -- activate LTB @--> {self.buy_threshold}")

            #Actual trading orders after price tracking activation
            #move target
        elif (self.up and self.current_price > self.record_price) or not self.up and self.current_price < self.record_price:
            self.record_price = self.current_price
            self.move_target()
            print(f'Updated target to {self.target}')
            sleep(3)

        #SELL
        elif self.up and self.current_price <= self.target:
            q = self.floor_to_n_digit(self.stash[self.bought], 1)
            price = self.floor_to_n_digit(self.target, 4)
            self.place_spot_order("SELL", self.pair, q, price)
            print(f"Just sold {q}, {self.pair}, at {self.current_price}")
            self.cooldown = (True, -1)
            self.deactivate()
            sleep(3)

        #BUY
        elif not self.up and self.current_price >= self.target:
            q = self.floor_to_n_digit(self.stash["USDC"]/self.target, 1)
            price = round(self.target, 4)
            self.place_spot_order("BUY", self.pair, q, price)
            print(f"Bought {q} {self.pair}")
            self.cooldown = (True, 1)
            self.deactivate()
            sleep(3)
        else:
            print(f"Current prce {self.current_price} within range, activation at {self.take_profit}:  Monitoring ...  ")
            sleep(3)

    def handle_ticker(self, msg):
        # Callback method Method that handles the tracking and logic. 
        try:
            if msg['e'] == '24hrTicker':
                self.current_price = float(msg['c'])
                self.up = not self.usdc
                self.check_oo()
                if self.freestyle:
                    ltp = self.transactions["price"]
                    if self.up :
                        self.look_to_sell(ltp*1.015)
                    else:
                        self.look_to_buy(ltp*0.985)

                #Logic alowing changing between trategies should also be added here.
                # For now choose the trading strategy by commenting or uncommenting the following lines 
                #self.interval_trading()
                #self.look_to_buy(0.6211)
                self.look_to_sell(0.6315)
                #self.api_calls()




        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Message received: {msg}")


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
        self.twm.start()
        self.twm.start_symbol_ticker_socket(callback=self.handle_ticker, symbol="ADAUSDC")
        print(f"Monitoring {self.pair} target at {self.take_profit} or {self.buy_threshold} up >> {self.up}, acttivated : {self.trigger}")

    def api_calls(self):
        #c = 0
        while True :
            try:
                import pdb

                pdb.set_trace()

                tick = self.cli.get_symbol_ticker(symbol=self.pair)
                self.current_price = round(float(tick['price']), 4)

            except:
                self.current_price = None

            if not self.current_price:
                while not self.current_price:
                    print("Trying to fetch the price")

                    tick = self.cli.get_symbol_ticker(symbol= self.pair)
                    self.current_price = round(float(tick['price']), 4)
            else : 

                self.up = not self.usdc

            # Calculate top and bottom prices from grid slopes
                print(f"Current  price {self.current_price} USD")
                self.current_timestamp = datetime.now().timestamp()

                # connstants or liniar variables
                self.buy_threshold, self.take_profit = (0.6421, 0.6975)#self.calculate_prices(self.current_timestamp)

                print(f'Activated: {self.trigger}\nactive between {self.buy_threshold} and {self.take_profit}\nGoing UP - {self.up}\n')
                c0 = self.cooldown[0]
                c05 = self.cooldown[1] > 0
                c1 = not self.trigger
                c2 = self.up and self.current_price >= self.take_profit # activates looking to sell
                c3 = not self.up and self.current_price <= self.buy_threshold # activates looking to buy 
                self.check_oo()

                #If on cooldown check temperature ---> check if price moves back into the intervall and mind the direction!
                #This is to prevent buying imediatly after selling when outside the intervall.
                if c0 :
                    print("On cooldown")
                    if c05 : # previously bought
                        if c2:
                            self.cooldown = (False, 0)
                            self.activate() #trigger in place ---> looking to sell
                            print("Activated: Looking to sell")
                    else:
                        if c3 :
                            self.cooldown = (False, 0)
                            self.activate() #trigger in place ---> looking to buy
                            print("Activated: Looking to buy")

                elif c1 and (c2 or c3):
                    self.activate()
                    print(f"Target designated {self.record_price}")

                elif c1 :
                    print("Price within range. Monioring ...")
                    if self.bought:
                        print(f"{self.current_price} --Activate LTS @--> {self.take_profit}")

                    else:
                        print(f"{self.current_price} -- activate LTB @--> {self.buy_threshold}")

                #move target
                elif (self.up and self.current_price > self.record_price) or not self.up and self.current_price < self.record_price:
                    self.record_price = self.current_price
                    self.move_target()
                    print(f'Updated target to {self.target}')

                #SELL
                elif self.up and self.current_price <= self.target:
                    q = self.floor_to_n_digit(self.stash[self.bought], 1)
                    price = self.floor_to_n_digit(self.target, 4)
                    self.place_spot_order("SELL", self.pair, q, price)
                    print(f"Just sold {q}, {self.pair}, at {self.current_price}")
                    self.cooldown = (True, -1)
                    self.deactivate()

                #BUY
                elif not self.up and self.current_price >= self.target:
                    q = self.floor_to_n_digit(self.stash["USDC"]/self.target, 1)
                    price = round(self.target, 4)
                    self.place_spot_order("BUY", self.pair, q, price)
                    print(f"Bought {q} {self.pair}")
                    self.cooldown = (True, 1)
                    self.deactivate()
                    sleep(3)
                else:
                    print(f"Current prce {self.current_price} within range, activation at {self.take_profit}:  Monitoring ...  ")
                    sleep(1)
            sleep(7)





    def stop_trading(self):
        self.twm.stop()
        print("Trading stopped.")

def main():
    tracker = PriceTracker(pair="ADAUSDC",
                           di1="2025-04-15 23:00:00",
                           df1="2025-04-21 22:00:00",
                           pi1=0.6021,
                           pf1=0.6399,
                           di2="2025-04-15 11:00:00",
                           df2="2025-04-25 15:00:00",
                           pi2=0.6315,
                           pf2=0.6315,
                           prc_distance= 0.00035)


    tracker.start_trading()
    #tracker.api_calls()

    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        pd.DataFrame(tracker.mock_transations).to_pickle("test_data.pkl")
        tracker.update_transactions_record(tracker.transactions)
        print("Manually stopped - data saved !")
        print(tracker.mock_usd, tracker.mock_mina)

    finally:
        tracker.stop_trading()

if __name__ == "__main__":
    main()
