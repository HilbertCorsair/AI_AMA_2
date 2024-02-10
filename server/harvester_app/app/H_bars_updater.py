from binance import Client

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = str(lines[0].splitlines()[0])
    b = str(lines[1])
    return Client(a , b)

latest_ts_file = "./data/h/last_update.txt"
files_to_update = ["./data/h/ADAUSDT_1h.csv",  "./data/h/BTCUSDT_1h.csv",  "./data/h/MINAUSDT_1h.csv", "./data/h/PAXGUSDT_1h.csv",
"./data/h/ADABTC_1h.csv", "./data/h/MINABTC_1h.csv",  "./data/h/PAXGBTC_1h.csv"  ]

# reads the last Hourly timestamp recorded
def get_latest_ts(file = latest_ts_file) :#"./data/latest_timestamp.txt", ):
    with open(file) as f:
        ts = int(f.readline())
    return ts

def update_coin_data_H(data_file ="./data/btc_2017H_bars.csv"):
    latest_ts = int(get_latest_ts())
    print(latest_ts)
    cli = unlock()
    new_df = cli.get_historical_klines('BTCUSDT', '1h', latest_ts)
    new_ts = new_df[-1][0]

    with open(data_file, 'a') as df:
        for line in new_df:
            df.write(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
        df.close()
        print("updated! check data files!")
        return new_ts

#update_coin_data_H()
nts = 0
for f in files_to_update:
    print(f'Updateing {f}...')
    nts = update_coin_data_H(f)
    print(f'Updated {f} to {nts}\n') 

# exporting latest timestamp
with open(latest_ts_file, "r+") as tf:
    tf.write(str(nts))
    tf.close
print (f"Latest timestamp: {nts} updated to >>> {latest_ts_file}")
