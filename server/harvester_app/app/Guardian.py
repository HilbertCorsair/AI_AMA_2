from AccountStatus import unlock
from time import sleep
#===================================================================================
of_interest =   ['BTC', 'ADA', 'MINA',    "PAXG", "AGIX", "DOT", "AR", "LINK"]
pairs = [f'{c}USDT' for c in of_interest]


BUSD_decs = [2, 4, 3, 0, 5, 2, 3, 2]
C_decs = [5, 1, 1, 4, 2, 3, 2, 3]

rounding_order_price = dict(zip(pairs, BUSD_decs))
rounding_order_crypro_amount = dict(zip(pairs, C_decs))

def get_price(pair):
    cli = unlock()
   # this gets the last traded price (no info on tranzaction type)
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, rounding_order_price[pair])
    return price


def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

def sell( pair, q , price):
    cli = unlock()
    try: 
        cli.create_order(
            symbol = pair,
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        print(f" Placing SELL {pair} order ...")
        #open_order = cli.get_open_orders(symbol = pair )
        while cli.get_open_orders(symbol = pair ):
            #orderID = open_order[0]['orderId']
            if c1 < 1 :
                print("waiting for order to be filled ...")
                c1 += 1
        
            else : 
                c1 += 1
                sleep(30)

        #print("DONE!")

    except BinanceAPIException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
def buy( pair, q , price):
    cli = unlock()
    try: 
        cli.create_order(
            symbol = pair,
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)
        
        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        print(f'Placeing BUY order...')
        #open_order = cli.get_open_orders(symbol = pair )
        while cli.get_open_orders(symbol = pair ):
            #orderID = open_order[0]['orderId']
            if c1 < 1 :
                print("waiting for order to be filled ...")
                c1 += 1
            else :
                c1 += 1
                sleep(30)
        #print("DONE!")

    except BinanceAPIException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)

from binance.enums import *
# Define the order parameters in a dictionary
order_params = {
    'symbol': 'MINABTC',  # Replace with the symbol you want to trade
    'side': 'SELL',  #BUY
    'type': 'STOP_LOSS_LIMIT',
    'timeInForce': 'GTC',
    'quantity': "400",    # Replace with the quantity you want to trade
    'price': '0.00002619',       # This is the limit price, replace with your price
    'stopPrice': '0.00002611'    # This is the stop price, replace with your stop price
}

# Place a STOP_LOSS_LIMIT order
def place_stop_loss (order_params_dict):
    cli = unlock()
    c1 = 0
    # While loop waits 50 min for order to fill than cancels 
    while cli.get_open_orders(symbol = order_params_dict['symbol'] ):
        #orderID = open_order[0]['orderId']
        if c1 < 1 :
            print("waiting for order to be filled ...")
            c1 += 1
        
        else : 
            c1 += 1
            sleep(30)

        #print("DONE!")
    cli.create_order(**order_params)
    print('Stopp loss placed')


def defend (pair, stop_level, price):
    """Takes a pair, a stop level a price ans a side. It monitors the price and places a stop loss order 
    This should be used when these are already buy or sell orders placed.
    The function waits for the order to be filled and then places the stop loss 
    """
    cli = unlock()
    active = True if cli.get_open_orders(symbol = pair) else False
    #1.0835
    while active:
        price = get_price(pair)

        if price <= stop_level:
            """
            order_params["symbol"] = pair
            order_params['quantity'] = "900"
            order_params['stopPrice'] = str(stop_level) #"1.0643"
            order_params['price'] = str(price) #"1.0605"
            order_params['side'] = side 
            place_stop_loss(order_params)
            active = False if cli.get_open_orders(symbol = pair) else True
            print(active)
            print(cli.get_open_orders(symbol = pair)[0]["orderId"])
            """
            #elif price <= 0.9983:
            # cancel open order
            oid = cli.get_open_orders(symbol = pair)[0]["orderId"]
            cli.cancel_order(
                symbol = pair,
                orderId = str(oid))
            print("canceled order")
            sell(pair, 900, 1.0115)
            print("Sold mina as safety precaution")
            active = True if cli.get_open_orders(symbol = pair) else False
        else:
            print(price)
            sleep(60*3)



def monitor(pair, stop_loss_price, acion_price):
    cli = unlock()
    price = get_price(pair)
    active = True if cli.get_open_orders(symbol = pair) else False

    while not active : 

        if price <= stop_loss_price:
            sell(pair, 1000, acion_price)
            print("Sold mina as safety precaution")
            active = True if cli.get_open_orders(symbol = pair) else False
        else:
            print(price)
            sleep(60*3)

monitor("MINAUSDT", 0.9693, 1.009)            
defend("MINAUSDT", 0.9283, 0.9501)
print("Defense protocole executed ! ")