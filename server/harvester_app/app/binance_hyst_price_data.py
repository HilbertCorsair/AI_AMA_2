from binance import Client
import pandas as pd

of_interest =  ["SOL", "PIXEL", "JUP", "WIF", "PYTH"]#['BTC', 'ADA', 'MINA', "PAXG", "AGIX", "DOT", "AR", "LINK"]

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)
cli = unlock()

ind = input("Select:\n1) BTC\n2) ADA\n3) MINA\n4) PAXG\n5) AGIX\n6) DOT\n7) all of the above\n")
ind = int(ind)
if ind == 7:
    cli = unlock()
    period = input("Select 'h' for hourly candles or 'd' for dayly candles or 'w' for weekly:\n")
    for coin in of_interest:
        #for ref in ["BUSD", "BTC"]:
        pair = f'{coin}USDT'
            #if not pair == "BTCBTC":
        print(f'Pair: {pair}\n')
        candle = f'1{period}'
        timestamp = cli._get_earliest_valid_timestamp(pair, candle)
        date = pd.to_datetime(timestamp, unit='ms', infer_datetime_format=True) 
        print(date)

        # gets historical HOURLY data back to timestamp in chunks of 1000 poins per call 
        print('Getting historical data ... ')
        bars = cli.get_historical_klines(pair, candle, timestamp, limit=1000)
        #with open('./data/btc_2017d_bars.json', 'w') as f:
        #    json.dump(bars, f)
        pth = f'../data/{period}/{pair}.csv'
        with open(pth, 'w') as d:
            for line in bars:
                d.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
    print("all done, check data files")
else:
    ind -=1
    period = input("Select 'h' for hourly candles or 'd' for dayly candles or 'w' for weekly:\n")
    choice = of_interest[ind]

    pair = f'{choice}USDT'
    candle = f'1{period}'
    cli = unlock()

    # get data from binance : choose period from : 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    timestamp = cli._get_earliest_valid_timestamp(pair, candle)
    date = pd.to_datetime(timestamp, unit='ms', infer_datetime_format=True) 
    print(date)

    # gets historical HOURLY data back to timestamp in chunks of 1000 poins per call 
    print('Getting historical data ... ')
    bars = cli.get_historical_klines(pair, candle, timestamp, limit=1000)
    #with open('./data/btc_2017d_bars.json', 'w') as f:
    #    json.dump(bars, f)
    pth = f'./data/{period}/{pair}_{candle}.csv'
    with open(pth, 'w') as d:
        for line in bars:
            d.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
    print("all done, check data files")

 
