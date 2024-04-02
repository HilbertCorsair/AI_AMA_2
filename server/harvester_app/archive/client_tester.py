from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from time import sleep





of_interest = ['BTC', 'ADA', 'MINA', "PAXG", "BUSD"]
pairs = ["ADABUSD" ,"BTCBUSD", "MINABUSD", "PAXGBUSD", "ADABTC", "MINABTC", "PAXGBTC"]
BUSD_decs = [3, 1, 3, 1, 7, 7, 5]
rounding_fiat = dict(zip(pairs, BUSD_decs))
C_decs = [1, 4, 1, 2, 6, 6, 3]
soldrounding_order_crypro_amount = dict(zip(pairs, C_decs))


def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

def get_price(pair):
    candles = cli.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_30MINUTE)
    lc= candles[-1]
    df = pd.DataFrame( columns= ['date', 'open', 'high', 'low', 'close'])
    df.loc[len(df.index)] = [float(x) for x in lc[0:5]]
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    avg = float((df["open"] + df["close"]) / 2)
    r = rounding_fiat[pair]
    avg = round(avg, r)
    return avg


# create a real order if the test orders did not raise an exception

cli = unlock()
#open = cli.get_open_orders()
""" 
def get_holding(stash_dict):
    prices = [get_price(f'{coin}BUSD') for coin in of_interest[0:4]]
    prices = dict(zip(of_interest[0:4], prices ) )
    stash_val = {}

    for k,v in prices.items():
        stash_val[k] = v * stash_dict[k]
    stash_val["BUSD"] = stash_dict['BUSD']
    max_key = max(stash_val, key=stash.get)
    return max_key



balance = [cli.get_asset_balance(c)for c in of_interest]
nl = ['BTC', 'ADA', 'MINA', "PAXG"]
stash = dict(zip(of_interest, [float(d["free"]) for d in balance]))
def buy_limit_order (coin, price, f = 1, ):
    fund = int (stash["BUSD"])*f
    q = round(fund/price) 
    cli.order_limit_buy(symbol=coin, quantity = q, price = price)


pairs = ["ADABUSD" ,"BTCBUSD", "MINABUSD", "PAXGBUSD", "ADABTC", "MINABTC", "PAXGBTC"]
decs = [3, 1, 3, 1, 6, 6, 4]
rounding = dict(zip(pairs, decs))

def buy( pair, q , price):
    cli = unlock()
    try: 
        buy_limit = cli.create_order(
            symbol = pair,
            side='buy',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        
        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
"""
""" 
        while cli.get_open_orders(symbol = pair ) :
            print(f"{c1}, Waiting for BUY {pair} order to execute{ord} ...")
            c1 += 1
            if c1 == 50:
                cli.cancel_order(symbol = pair, orderID = buy_limit['orderId'])
                break

            else : 
                sleep(15)
"""
""" 
    except BinanceAPIException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)

def sell( pair, q , price):
    cli = unlock()
    try: 
        sell_limit = cli.create_order(
            symbol = pair,
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        #ord = cli.get_open_orders(symbol = pair )
        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        while cli.get_open_orders(symbol = pair ):
            print(f"{c1}, Waiting for BUY {pair} order to execute ...")
            c1 += 1
            if c1 == 50:
                cli.cancel_order(symbol = pair, orderID = sell_limit['orderId'])
                break

            else : 
                sleep(15)
        print("It might have worked")

    except BinanceAPIException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)


print(cli.get_asset_balance(asset = "ADA"))
latest_price = cli.get_symbol_ticker(symbol = "ADABUSD")

cli.create_order(
            symbol = "ADABUSD",
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity = 50,
            price = 0.35)

oor = cli.get_open_orders(symbol = 'ADABUSD' )
id1 = oor[0]['orderId']
"""
 
#cli.cancel_order(
#    symbol='ADABUSD',
#    orderId= id1)
#print(cli.get_open_orders(symbol = 'ADABUSD' ))

#print(cli.get_open_orders(symbol = "MINAUSDT"))
a = cli.get_open_orders(symbol = "MINABTC") 
b = cli.get_account()
c = cli.get_all_orders()
print(2) if a else print(1)
print(c)