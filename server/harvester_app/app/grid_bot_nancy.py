#!/usr/bin/python3
from binance import Client
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException

bottom_lim = 15700
top_lim = 18335

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = str(lines[0].splitlines()[0])
    b = str(lines[1])
    return Client(a , b)

cli = unlock()

btc_ballance = float(cli.get_asset_balance(asset='BTC')["free"])
busd_ballance = float(cli.get_asset_balance(asset='BUSD')["free"])

counter = 0
while counter < 5:

    btc_price = float(cli.get_symbol_ticker(symbol="BTCUSDT")["price"])

    if btc_price <= bottom_lim and  busd_ballance > 100 :
        try: 
            
            buy_order_limit = cli.create_order(
            symbol='BTCUSDT',
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity = (busd_ballance * 0.8)/btc_price,
            price = bottom_lim)

            counter +=1 

        except BinanceAPIException as e:
            # error handling goes here
            print(e)
        except BinanceOrderException as e:
            # error handling goes here
             print(e)


        # if price reaches TOP limit or goes above SELL
    elif btc_price >= top_lim and  btc_ballance > 0.001 :
        try: 
            
            sell_order_limit = cli.create_order(
            symbol='BTCBUSD',
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity= btc_ballance * 0.8 ,
            price = top_lim)

            counter +=1 

        except BinanceAPIException as e:
            # error handling goes here
            print(e)
        except BinanceOrderException as e:
            # error handling goes here
            print(e)
        
    else:
        print(f'BTC price is {btc_price}')
        sleep(30)
        counter += 1 




    

    

        
            

