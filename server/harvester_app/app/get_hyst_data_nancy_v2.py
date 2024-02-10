from binance import Client
import pandas as pd

#of_interest = [ "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "SHIB", "LINK", "AR", "AAVE", "EGLD"]
of_interest =  ["KSM", "ENJ"]#["ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "COTY",
delta = [
    69020 - 15460,
    3.1 - 0.239,
    6.68 - 0.24,
    2070 - 1606,
    0.95 - 0.036,
    55.13 - 4,
    2.95 - 1.15,
    693 - 183,
    2.94 - 0.315,
    0.00008872 - 0.00001706,
    53.1 - 5,
    91.24 -6,
    666.7- 45.6,
    544.7 - 32.31
    ]

intervals = dict(zip(of_interest, delta))

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)
cli = unlock()

timestamp = 1665734400000
for coin in of_interest:
    pair = f'{coin}BUSD'
    
    print(f'Pair: {pair}\n')
    candle = '1h'
    #timestamp = cli._get_earliest_valid_timestamp(pair, candle)
    #print(pair , timestamp)
    
    #date = pd.to_datetime(timestamp, unit='ms', infer_datetime_format=True) 
    #print(date)

    # gets historical HOURLY data back to timestamp in chunks of 1000 poins per call 
    #print(f'Getting historical data for {pair}... ')


    bars = cli.get_historical_klines(pair, candle, timestamp, limit=1000)
    #with open('./data/btc_2017d_bars.json', 'w') as f:
    #    json.dump(bars, f)
    pth = f'./data/h/{pair}_{candle}2.csv'
    with open(pth, 'w') as d:
        for line in bars:
            d.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
print("all done, check data files")
