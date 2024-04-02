import numpy as np
import schedule
from time import sleep
import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager
import btalib as bl
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas_ta as ta

coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]

mins = [
    15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
    869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230, 1.1
    ]
tops = [
    69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
    4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85, 19.23
    ]

mins_dict = dict(zip(coins, mins))
tops_dict = dict(zip(coins, tops))

of_interest =   ['BTC', 'ADA', 'MINA',    "PAXG", "AGIX", "DOT", "AR", "LINK"]
pairs = [f'{c}USDT' for c in of_interest]


BUSD_decs = [2, 4, 3, 0, 5, 2, 3, 2]
C_decs = [5, 1, 1, 4, 0, 3, 2, 3]

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
cli = unlock()

def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def norm_augment(df): 
    #df = df.drop_duplicates().copy()
    # Assuming df is a pandas DataFrame with 'high', 'low', 'open', 'close' columns.
    top_price = df['high'].max()
    bottom_price = df['low'].min()

    # Vectorized operations for efficiency
    df['avg'] = (df['high'] + df['low']) / 2
    df['open'] = (df['open'] - bottom_price) / (top_price - bottom_price)
    df['high'] = (df['high'] - bottom_price) / (top_price - bottom_price)
    df['low'] = (df['low'] - bottom_price) / (top_price - bottom_price)
    df['close'] = (df['close'] - bottom_price) / (top_price - bottom_price)
    df = df.loc[~df.index.duplicated(keep='first')]

    # Calculate MACD using pandas_ta
    macd = df.ta.macd(close='close', fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)

    # Limit df to the last 1000 entries
    df.dropna(inplace=True)
    

    # Convert index to matplotlib date numbers if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        x = np.array([date2num(d) for d in df.index])
    else:
        x = np.arange(len(df))

    y = df['MACDh_12_26_9'].values
    # Momentum calculation
    df['momentum'] = np.gradient(y, x)

    # LOWESS smoothing
    #lowess = sm.nonparametric.lowess(y, x, frac=0.019)
    #df['LOWESS'] = lowess[:, 1]


    return df

# Updating the Coins data dictionary
def update_pair (pair, timestamp):
    cli = unlock()
    bars = cli.get_historical_klines(pair, "1h", timestamp)
    #create empty dataframe
    df = pd.DataFrame( columns= ['date', 'open', 'high', 'low', 'close'])
    for bar in bars:
        # add each row to the dataframe
        df.loc[len(df.index)] = [float(x) for x in bar[0:5]]
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def get_price(pair):
    cli = unlock()
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, rounding_order_price[pair])
    return price

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

#==========================================================================================================
#data_files =  [f'/home/honeybadger/projects/harvester/data/h/{pair}.csv'for pair in pairs ] 
#c_data = [import_coin_data(pth) for pth in data_files] #from old csvs !!! ALSO ADDS DATES AS INDEXES


# # UPDATE FILES AND REWRITE the pkl files
def update_h_candles(pairs):
    c_data = [pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl").tail(468) for p in pairs]
    c_dict = dict(zip(pairs, c_data))
    
    for p in pairs:
        timestamp =  int(c_dict[p].index[-1].timestamp())*1000
        # This part is useless if data in c_dict is up to date
        new_df = update_pair(p, timestamp).iloc[1 : ]
        
        if not new_df.empty:
            # Add data to existing dict and also save updatated to pkl
            c_dict[p] = pd.concat([c_dict[p], new_df], axis=0)
            c_dict[p].to_pickle(path = f"/home/honeybadger/projects/harvester/data/h/pkls/{p}.pkl")
            print(f"{p} updated")
        else:
            print("No new data.")
    print("Update complete!")
    
    for pair, df in c_dict.items():
        c_dict[pair] = norm_augment(df)
        cols = c_dict[pair].columns   
        c_dict[pair].columns = [f'{pair}_{col}' for col in cols ]

    return c_dict


# Schedule the process function to run every hour
def process():
    c_dict = update_h_candles(pairs)
    prices_dict = dict(zip(pairs, [get_price(pair) for pair in pairs]))
    #buy = []
    #sell = []
    lbls = of_interest
    record = {}
    for c in coins:
        record[c] = pd.DataFrame( columns = ["sign", 'hyst'])

    balance = [cli.get_asset_balance(c)for c in lbls + ["USDT"]]
    
    stash = dict(zip(lbls + ["USDT"], [float(d["free"]) for d in balance if d["asset"] in lbls + ["USDT"]]))
    # Determining the holding position
    valuations = {}
    for k, val in prices_dict.items():
        coin = [c for c in lbls if c in k][0]
        valuations[coin] = val * stash[coin]
    valuations["USDT"] = stash["USDT"]
    valuations = pd.Series(valuations)
    
    bought = valuations.idxmax()

    bags = {"fiat": 100}
    for c in lbls:
        bags[c]= 0

    market_dict = {}
    for p in pairs:
        hist = c_dict[p][f"{p}_MACDh_12_26_9"].iloc[-1]
        momentum =c_dict[p][f"{p}_momentum"].iloc[-1]
        pi =c_dict[p][f"{p}_avg"].iloc[0]
        pf = c_dict[p][f"{p}_avg"].iloc[-1]
        mm = round((pf - pi)/pi, 2 )     
        print(f"{p}: {round(hist, 5)}: {round(momentum, 4)}, {round(pi,2)} --> {round(pf, 2)} | {mm}")
        market_dict[p] = (c_dict[p][f'{p}_avg'].iloc[-1] - c_dict[p][f'{p}_avg'].iloc[0] ) /c_dict[p][f'{p}_avg'].iloc[0]
    
    avg_mkt = sum(market_dict.values())/len(pairs)
    print(f"Average market move: {round(avg_mkt*100, 2)}" )

    big_df = pd.concat(c_dict.values(), axis= 1)

    
    for ind in big_df.index :
        #ind2 = df.index[i+ 1] if not i > 2653  else df.index[i]
        #print(ind)
        #day = weekDays[ind.weekday()]

        histos = big_df.loc[ind, : ].filter(like = "MACDh_12_26_9" ) == "USDT" != "USDT"
        histos.sort_values(ascending=False, inplace= True)
    
        momentae = big_df.loc[ind, : ].filter(like = "momentum" )
        momentae.sort_values(ascending = False, inplace = True)
        # take snapshot of the amplitude and direction of the price movement 
        
        pair = momentae.idxmax().split("_")[0]
        #pair_hist = histos[0]
        vector  = momentae.loc[f"{pair}_momentum"]

        if  vector > 0.01 and bought == "USDT":
            #pair = snapshot["hist_max_idx"].split("_")[0]
            # BUY !
            #place buy order
            bought = [str for str in lbls if str in pair][0]
    
            bags[bought] = bags["fiat"] / big_df.loc[ind, f'{bought}USDT_avg']
            bags["fiat"] = 0
            #print(f"4. Bought {bought}")

            td = {"date": [ind], "sign": [1], "hyst": [big_df.loc[ind, f'{bought}USDT_MACDh_12_26_9']]}
            store = pd.DataFrame(td)
            store.set_index("date", inplace= True)
            record[bought] = pd.concat([record[bought], store], axis=0)
            q = round ((stash["USDT"]-5)/prices_dict[pair], rounding_order_crypro_amount[pair])
            p = round(prices_dict[pair], rounding_order_price[pair])
            print(f'buying {q} of {pair} for {p}')
            buy(pair, q, p)

        elif vector < 0 and bought != "USDT":
            #SELL
            bags["fiat"] = bags[bought] * big_df.loc[ind, f'{bought}USDT_avg']
            #print(f'2. Sold {pair}: bag is now {bags["fiat"]}$ ')


            td = {"date": [ind], "sign": [-1], "hyst": [big_df.loc[ind, f'{bought}USDT_MACDh_12_26_9']]}
            store = pd.DataFrame(td)
            store.set_index("date", inplace= True)
            record[bought] = pd.concat([record[bought], store], axis=0 )

            bags[bought] = 0
            bought = False
            q = round(stash[bought], rounding_order_crypro_amount[f'{bought}USDT'])
            p = round(prices_dict[bought], rounding_order_price[f'{bought}USDT'])
            sell(pair, q, p)


        else:
            pass

                
    if bags["fiat"] == 0 :
        bags["fiat"] = bags[bought] * big_df.loc[ind, f'{bought}USDT_avg']
        bags[bought] = 0

    print( bags["fiat"] )




    fig = go.Figure()
    for p, df in c_dict.items():
        fig.add_trace(go.Scatter(x=c_dict[p].index, y=c_dict[p][f'{p}_MACDh_12_26_9'].values, mode='lines', name=p))

    fig.show() 

#plt.plot(c_dict["BTCUSDT"].index, c_dict['BTCUSDT']["BTCUSDT_histogram"].values)

# Adding labels at the start of each line
#for p  in pairs :
#    plt.plot(c_dict[p].index, c_dict[p][f'{p}_histogram'].values, label = p)
#    #plt.text(c_dict[p].index[0], c_dict[p][f'{p}_histogram'][0], p , fontsize=9, verticalalignment='bottom')

# Additional plot formatting
#plt.xlabel('time (h)')
#plt.ylabel('macnd hyst')
#plt.title('coins of interest')
#plt.legend()
#plt.show()


stats_dict = {}
for p in pairs:
    data = c_dict[p].tail(33)[[f"{p}_high", f"{p}_low"]]
    overall_mean = data.values.flatten().mean()
    overall_std = data.values.flatten().std()
    # mean and sd for overall highs and lows of the last 33 hours
    stats_dict[p] = (overall_mean, overall_std)


print(stats_dict["MINAUSDT"])




