from binance import Client
import oerations as ops
def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

cli = unlock()

ballance = cli.get_account()
all_binance = ballance['balances']
binance_holdings = [x for x in all_binance if float(x['free']) > 0.09 or float(x['locked']) > 0.09]
available = [x for x in all_binance if float(x["free"])  > 0]
locked = [(x["asset"], float(x["locked"])) for x in all_binance if float(x['locked']) > 0 ]

coins = cli.get_symbol_ticker()
symbols = [c["symbol"] for c in coins]

binance_usdt_balance = [float(asset['free']) +
                    float(asset['locked'])
                    for asset in binance_holdings
                    if asset['asset'] == 'USDT']

for hold in binance_holdings:
    sy = hold["asset"]
    tk = str(f'{sy}USDT')
    x = tk in symbols
    
    if x :
        hol = float(hold['free']) + float(hold['locked'])
        x_tk = cli.get_symbol_ticker( symbol = tk)
        pri = float(x_tk['price']) 
        binance_usdt_balance.append(hol * pri)
        
usdt_value = sum(binance_usdt_balance)


if locked:
    locked = dict(locked)
    tvl = 0
    prices = dict([ (a,ops.get_price([f"{a}USDT"])) for a in list(locked.keys())])
    for k, v in locked.items():
        tvl += locked[k]
        print (f'{round(tvl/usdt_value, 2)} of total {usdt_value} is locked in {k}')
        exit()
    
    orders = {}
    for a in locked.keys():
        if cli.get_open_orders(symbol = f"{a}USDT"):
            orders[f"{a}USDT"] = cli.get_open_orders(symbol = f"{a}USDT")[0]["orderId"]

        if cli.get_open_orders(symbol = f"{a}BTC"):
            orders[f"{a}BTC"] = cli.get_open_orders(symbol = f"{a}BTC")[0]["orderId"]




