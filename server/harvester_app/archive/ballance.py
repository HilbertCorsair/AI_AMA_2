from binance import Client
import pandas as pd
import re

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

cli = unlock()



ballance = cli.get_account()
all_binance = ballance['balances']
binance_holdings = [(x) for x in all_binance if float(x['free']) > 0.09 or float(x['locked']) > 0.09]

#print(binance_holdings)

#client.get_symbol_ticker(symbol="BTCUSDT")


coins = cli.get_symbol_ticker()
symbols = [c["symbol"] for c in coins]


Bin_usdt_balance = [float(asset['free']) +
                    float(asset['locked'])
                    for asset in binance_holdings
                    if asset['asset'] == 'USDT']


Bin_usdt_balance = Bin_usdt_balance[0] if Bin_usdt_balance else 0
#Bin_usdt_balance = 0

usdt_conv = [Bin_usdt_balance]
for hold in binance_holdings:
    sy = hold["asset"]
    tk = str(f'{sy}USDT')
    x = tk in symbols

    if x :
        hol = float(hold['free'])
        x_tk = cli.get_symbol_ticker( symbol = tk)
        pri = float(x_tk['price']) 
        usdt_conv.append(hol * pri)
        
        
#Ada =  float(cli.get_symbol_ticker( symbol = 'ADAUSDT')["price"]) *1000
#mina = float(cli.get_symbol_ticker( symbol = 'MINAUSDT')["price"]) *2000
#usdt_conv.append(Ada)
#usdt_conv.append(mina)
#usdt_conv.append(710)
#usdt_conv.append(2180) # ERG value from GATE io
usdt_value = sum(usdt_conv)
print(round(usdt_value, 2))

