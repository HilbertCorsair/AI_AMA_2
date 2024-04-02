import pandas as pd 
from binance import Client
import requests

coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]
pairs = [f'{coin}BUSD' for coin in coins]


#data_files = [f'./data/h/{pair}_1h2.csv' if not pair == 'ERGBUSD' else "./data/h/ERGUSTD_1h.csv" for pair in pairs]

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

def update_coin_data_file(pair):
    pth = f'./data/h/{pair}_1h2.csv'
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    lts = df.index[-1]
    cli = unlock()
    print(f'Updating {pair} file : {pth}')
    bars = cli.get_historical_klines(pair, "1h", str(lts), limit=1000)
    with open(pth, 'a') as d:
        for line in bars:
            if not line[0]==str(lts):
                d.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
    print(f'{pair} updated!\n')

def update_ERG_data_file(pth = './data/h/ERGUSTD_1h.csv'):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    lts = df.index[-1]
    host = "https://api.gateio.ws"
    prefix = "/api/v4"
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    url = '/spot/candlesticks'
    query_param = f'currency_pair=ERG_USDT&from={lts}&interval=1h'
    print(f'Updating ERG file : {pth}')
    r = requests.request('GET', host + prefix + url + "?" + query_param, headers=headers)
    with open(pth, 'a') as d:
        for line in r.json():
            if not (line == "message" or line == "label" or int(line[0]) == lts):
                try:
                    d.write(f'{line[0]}, {line[2]}, {line[3]}, {line[4]}, {line[5]}\n')
                except:
                    print(line)
            else:
                pass
    print('ERG file updated')

[update_coin_data_file(p) if not p == "ERGBUSD" else update_ERG_data_file() for p in pairs]
