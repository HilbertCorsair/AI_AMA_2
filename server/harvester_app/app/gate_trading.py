import gate_api
from gate_api.exceptions import ApiException, GateApiException
import time

# Defining the host is optional and defaults to https://api.gateio.ws/api/v4
# See configuration.py for a list of all supported configuration parameters.
# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.


def unlock_gate (fname = 'gatev4.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    config = gate_api.Configuration(
        host = "https://api.gateio.ws/api/v4",
        key = a,
        secret = b
    )
    api_client = gate_api.ApiClient(config)
    # Create an instance of the API class
    api_instance = gate_api.SpotApi(api_client)

    return api_client, api_instance


""" 

try:
    #api_response = api_instance.list_tickers(currency_pair="ERG_USDT")
    api_erg = api_instance.get_currency_pair(currency_pair="ERG_USDT") #info regarding trading pair
    erg_candles = api_instance.list_candlesticks(currency_pair="ERG_USDT", interval = "1h")
    print(erg_candles)
except GateApiException as ex:
    print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
except ApiException as e:
    print("Exception when calling SpotApi->list_tickers: %s\n" % e)

api_client , api_instance = unlock_gate()

ts = 1665734400
to1 = ts + 1000*60*60
now = time.time()
h1ksago = now - 1000*60*60

print(f'Ts is {(now - ts)/3600} hours ago')

start = ts
stop = ts + 3600000 # 1000 hours in seconds
try:
    #api_response = api_instance.list_tickers(currency_pair="ERG_USDT")
    #api_erg = api_instance.get_currency_pair(currency_pair="ERG_USDT") #info regarding trading pair
    erg_candles = api_instance.list_candlesticks(currency_pair="ERG_USDT", from = start, to = stop, interval = "1h")
    print(erg_candles)
except GateApiException as ex:
    print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
except ApiException as e:
    print("Exception when calling SpotApi->list_tickers: %s\n" % e)

ximport requests
ts = 1665734400
now = time.time()
stop = ts + 3600*997

host = "https://api.gateio.ws"
prefix = "/api/v4"
headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

url = '/spot/candlesticks'

bars = []
for i in range(4):
    stop = min((ts + 3600*997), now)
    query_param = f'currency_pair=ERG_USDT&from={ts}&to={stop}&interval=1h'
    r = requests.request('GET', host + prefix + url + "?" + query_param, headers=headers)
    [bars.append(line) for line in r.json()]
    ts += 3600*997

pth = f'./data/h/ERGUSTD_1h.csv'
with open(pth, 'a') as d:
    for line in bars:
        if not (line == "message" or line == "label"):
            try:
                d.write(f'{line[0]}, {line[2]}, {line[3]}, {line[4]}, {line[5]}\n')
            except:
                print(line)
        else:
            pass
"""


# Define the API client
api_client, spot_api = unlock_gate()

def get_btc_price():
    # Fetch the ticker information for the BTCUSDT pair
    ticker = spot_api.list_tickers(currency_pair='BTC_USDT')
    # The API returns a list of tickers, but since we requested only one pair, we can take the first element
    latest_price = ticker[0].last
    print(f"The latest price for BTCUSDT is: {latest_price}")
    return latest_price

currency_pair = "BTC_USDT"
amount = "0.01"  # The amount of BTC you want to buy
price = "30000"  # The price at which you want to buy 1 BTC

limit_dict = {
    "side": "",
    "pair":"", 
    "amount":"",
    "price": "",
    "type" : "limit" 
}

def place_limit_order(limit_dict):
    order = gate_api.Order(
        currency_pair=limit_dict["pair"].replace("_", ""), # Ensure the pair format is correct, e.g., "BTCUSDT" instead of "BTC_USDT"
        side=limit_dict["side"], # "buy" or "sell"
        amount=str(limit_dict["amount"]),
        price=str(limit_dict["price"]),
        time_in_force="gtc", # Good Till Cancel
        type=limit_dict["type"] # "limit"
    )
    try:
        # Place the limit order
        response = spot_api.create_order(order)
        print(f"Order placed successfully: {response}")
        return response
    except ApiException as e:
        print(f"An error occurred while placing the order: {e}")
        return None

holding = 5
def limit_general(buy_target, sell_target):
    cp = get_btc_price()
    if cp <= buy_target and USDT_holding >= 10 :
        limit_dict = {
            "side": "buy",
            "pair":"",
            "amount":"",
            "price": "",
            "type" : "limit"
            } 
    elif cp >= sell_target and holding >= 10 :
        limit_dict = {
            "side": "sell",
            "pair":"",
            "amount":"",
            "price": "",
            "type" : "limit" 
            }
    else:
        print("No action!")
        

    
