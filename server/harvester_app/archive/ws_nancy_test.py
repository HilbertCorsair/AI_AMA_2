import os
from time import sleep
import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager

of_interest = ['BTC', 'ADA', 'MINA', "PAXG", "BUSD"]
pairs = ["ADABUSD" ,"BTCBUSD", "MINABUSD", "PAXGBUSD", "ADABTC", "MINABTC", "PAXGBTC"]


def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

price = {"ADABUSD": None, 'error': False}

def pairs_trade(msg):
    # define how to process incoming WebSocket messages
    if msg['e'] != 'error':
        price["ADABUSD"] = float(msg['c'])
    else:
        price['error'] = True

bsm = ThreadedWebsocketManager()
bsm.start()
bsm.start_symbol_ticker_socket(symbol="ADABUSD", callback= pairs_trade)
while not price["ADABUSD"]:
    sleep(0.1)

while True:
    if price['error']:
        bsm.stop()
        sleep(3)
        bsm.start()
        price["error"] = False
    else:
        if price['BTCUSDT'] > 10000:
            try :
                order = cli.order_market_buy(symbol="ADABUSD", quantity=100)
                break
            except Exception as e:
                print(e)
    sleep(0.1)


    