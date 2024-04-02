from tracemalloc import stop
import pandas as pd
import cbpro as cb

# public data  
c_public = cb.PublicClient(api_url= "https://api-public.sandbox.exchange.coinbase.com" )
data = pd.DataFrame(c_public.get_products())
print(data.tail().T)
"""
#get ticker
import requests

# real exchange APII link : "https://api.pro.coinbase.com/products/ADA-USD/ticker"
ticker = requests.get('https://api-public.sandbox.exchange.coinbase.com/products/BTC-USD/ticker').json()

#get hystorical data
historical = pd.DataFrame(c_public.get_product_historic_rates(product_id='ADA-USD'))
historical.columns= ["Date","Open","High","Low","Close","Volume"]
historical['Date'] = pd.to_datetime(historical['Date'], unit='s')
historical.set_index('Date', inplace=True)
historical.sort_values(by='Date', ascending=True, inplace=True)
 
print("\n",  historical , "\n")

historical['15 SMA'] = historical.Close.rolling(15).mean()
historical['SD'] = historical.Close.rolling(15).std()

print (historical.tail())


#plotting the candle chart of the historical data

import plotly.graph_objects as go
fig = go.Figure(data=[go.Candlestick(x = historical.index,
                                    open = historical['Open'],
                                    high = historical['High'],
                                    low = historical['Low'],
                                    close = historical['Close'],
                                    ),
                     go.Scatter(x=historical.index, y=historical['15 SMA'], line=dict(color='purple', width=1))])


fig.show()

# getting order book
order_book = c_public.get_product_order_book('ADA-USD') #json
bids = pd.DataFrame(order_book['bids'])
asks = pd.DataFrame(order_book['asks'])

df = pd.merge(bids, asks, left_index=True, right_index=True)
df = df.rename({"0_x":"Bid Price","1_x":"Bid Size", "2_x":"Bid Amount",
                "0_y":"Ask Price","1_y":"Ask Size", "2_y":"Ask Amount"}, axis='columns')
print(df.head())

# getting trade data
trades = pd.DataFrame(requests.get('https://api.pro.coinbase.com/products/ETH-USD/trades').json())
print (trades.tail())

# web sockets


wsc = cb.WebsocketClient(url="wss://ws-feed.pro.coinbase.com",
                                products="ADA-USD",
                                channels=["ticker"])
wsc.close()

# new class for web sockets
import time

class ADAsocket(cb.WebsocketClient):
    def on_open(self):
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.products = ["ADA-USDT"]
        self.channels=["ticker"]
        self.message_count = 0
    def on_message(self, msg):
        self.message_count += 1
        print(msg)
        print(msg['type'])
        if 'price' in msg and 'type' in msg:
            print("here we go!")
            print ("Message type:", msg["type"],
                   "\t@ {:.3f}".format(float(msg["price"])))
    def on_close(self):
        print("Closing")

wsClient = ADAsocket()
wsClient.start()
dans_counter = 0
print(wsClient.url, wsClient.products, wsClient.channels)
while (wsClient.message_count < 50):
    print("---> ", dans_counter)
    dans_counter += 1
    print ("\nmessage_count =", "{} \n".format(wsClient.message_count))
    time.sleep(1)
wsClient.close()   


"""
# Actual trading on a target price
import base64
import json
from time import sleep

# sandbox websocket
# cb.WebsocketClient(url="wss://ws-feed-public.sandbox.exchange.coinbase.com")


key = '978638962982ec0fda25a57fdd55e4c6'
secret = 'Y+afeGgiZgHluJ4J0dR5NAjWNfuqsUdCdSjzTcaXmBo29rVIF3AiV3aDNHGN6RrS2FyJvLNJZYN98V9q1lVp9Q=='
passphrase = 'yb1sky8j2q'

encoded = json.dumps(secret).encode()
b64secret = base64.b64encode(encoded)
auth_client = cb.AuthenticatedClient(api_url= "https://api-public.sandbox.exchange.coinbase.com", key=key, b64secret=secret, passphrase=passphrase)

#  c_public

import time
while True:
    try:
        ticker = c_public.get_product_ticker(product_id='BTC-USD')
    except Exception as e:
        print(f'Error obtaining ticker data: {e}')
    
    if float(ticker['price']) >= 22000.00:
        try:
            limit = c_public.get_product_ticker(product_id='ETH-USD')
        except Exception as e:
            print(f'Error obtaining ticker data: {e}')
        
        try:
            print("HERE WE GOO ---> ",ticker['price'])
            order=auth_client.place_limit_order(product_id='BTC-USDT', 
                              side='buy', 
                              price=22100, 
                              size='0.007')
            print(order['id'])

        except Exception as e:
            print(f'Error placing order: {dir(e)}\n >>{e.args} \n>>{e.__cause__}\n>>>>{e.with_traceback}\n>>>>{e.__class__}')
        
        sleep(2)
        
        try:
            check = order['id']
            check_order = auth_client.get_order(order_id=check)
        except Exception as e:
            print(f'Unable to check order. It might be rejected. {e}')
        
        if check_order['status'] == 'done':
            print('Order placed successfully')
            print(check_order)
            break
        else:
            print('Order was not matched')
            break
    else:
        print(f'The requirement is not reached. The ticker price is at {ticker["price"]}')
        sleep(10)
