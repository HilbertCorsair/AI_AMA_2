import requests
import time
import hmac
import hashlib
import base64
import json
from kucoin.client import Client
from kucoin.client import Client
client = Client(api_key, api_secret, api_passphrase)
def unlock (fname = 'kuk2.txt'):
    """Returns the Binance Client"""
    with open(fname) as f:
        lines = f.readlines()
    b = lines[1].splitlines()[0]
    c = lines[2].splitlines()[0]
    d = lines [3].splitlines()[0]

    return Client(c,d,b)

cli = unlock()


def sign_request(endpoint, method, body=''):
    now = int(time.time() * 1000)
    str_to_sign = str(now) + method + endpoint + body
    signature = base64.b64encode(hmac.new(key[2].encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
    passphrase = base64.b64encode(hmac.new(key[2].encode('utf-8'), "strada1848".encode('utf-8'), hashlib.sha256).digest())
    headers = {
        "KC-API-KEY": key[1],
        "KC-API-SIGN": signature.decode(),
        "KC-API-TIMESTAMP": str(now),
        "KC-API-PASSPHRASE": passphrase.decode(),
        "Content-Type": "application/json"
    }
    return headers

def order_params (side, price, quantity, symbol= "ERG-USDT", type = "limit" ):
    order_dict = {
        "clientOid": str(int(time.time() * 1000)),
        "side": side,
        "symbol": symbol,
        "type": type,
        "price": price,
        "size": quantity
    }
    return order_dict

def place_lim_order(params):
    endpoint = "/api/v1/orders"
    url = "https://api.kucoin.com" + endpoint
    method = "POST"
    body = json.dumps(params)
    headers = sign_request(endpoint, method, body)
    response = requests.post(url, headers=headers, data=body)
    return response.json()

op = order_params("buy", 0.9511, 200)


def fetch_order_book():
    url = "https://api.kucoin.com/api/v1/market/orderbook/level2_20?symbol=ERG-USDT"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()
        if data['code'] == '200000':
            order_book = data['data']
            print_order_book(order_book)
        else:
            print(f"Error fetching order book: {data['msg']}")
    except requests.exceptions.RequestException as e:
        print(f"HTTP request failed: {e}")

def sign_request(endpoint, method, body='', query=''):
    now = int(time.time() * 1000)
    str_to_sign = str(now) + method + endpoint + query + body
    signature = base64.b64encode(
        hmac.new("0285db49-f2f3-4141-bdb7-6a929ec6d9e4".encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest()
    )
    headers = {
        "KC-API-KEY": "667499ccf323d000014e309a",
        "KC-API-SIGN": "0285db49-f2f3-4141-bdb7-6a929ec6d9e4",
        "KC-API-TIMESTAMP": str(now),
        "KC-API-PASSPHRASE": "strada1848",  # Passphrase should be sent as plain text
        "KC-API-KEY-VERSION": "3",
        "Content-Type": "application/json"
    }
    return headers

def check_active_order(symbol):
    endpoint = "/api/v1/orders"
    url = "https://api.kucoin.com" + endpoint
    method = "GET"
    query = f"?symbol={symbol}&status=active"
    headers = sign_request(endpoint, method, query=query)
    response = requests.get(url + query, headers=headers)
    data = response.json()
    
    if data['code'] == '200000':
        orders = data['data']['items']
        if orders:
            print(f"Active orders for {symbol}:")
            for order in orders:
                print(order)
            return orders
        else:
            print(f"No active orders for {symbol}.")
            return None
    else:
        print(f"Error fetching active orders: {data['msg']}")
        return None



def print_order_book(order_book):
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    total_buy_volume = sum(float(bid[1]) for bid in bids)
    total_sell_volume = sum(float(ask[1]) for ask in asks)

    print("\nOrder Book for ERG/USDT:")
    print("Bids (Buy Orders):")
    for bid in bids[:5]:  # Print top 5 bids
        print(f"Price: {bid[0]}, Volume: {bid[1]}")

    print("Asks (Sell Orders):")
    for ask in asks[:5]:  # Print top 5 asks
        print(f"Price: {ask[0]}, Volume: {ask[1]}")

    print(f"Total Buy Volume: {total_buy_volume}")
    print(f"Total Sell Volume: {total_sell_volume}")

if __name__ == "__main__":
    fetch_order_book()
    place_lim_order(op)
    check_active_order(symbol="ERG-USDT")

