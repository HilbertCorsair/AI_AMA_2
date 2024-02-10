import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import seaborn as sb
from sys import exit
from binance import Client
from datetime import datetime
import math


coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]
pairs = [f'{coin}BUSD' for coin in coins]

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

data_files = [f'./data/h/{pair}_1h2.csv' if not pair == 'ERGBUSD' else "./data/h/ERGUSTD_1h.csv" for pair in pairs]
#coins = ["PAXG", "AGIX", "KAVA", "FIL", "RNDR", "MATIC", "BTC", "ALGO", "MINA"]
pairs = [f'{coin}BUSD' for coin in coins]
#data_files = [f'./data/h/{pair}_1h2.csv' if not pair == 'ERGBUSD' else "./data/h/ERGUSTD_1h.csv" for pair in pairs]
#data_files =  [f'./data/h/{c}BUSD_1h2.csv'for c in coins]

def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def delta (arr):
    i = arr[0] #if abs(arr[0]) < 0.15 else 0
    f = arr[-1] # if abs(arr[0]) < 0.15 else 0

    cases_sign = [
        i > 0 and f > 0,# 0
        i > 0 and f < 0,# 1
        i < 0 and f > 0,# 2
        i < 0 and f < 0 # 3
        ]
    
    if abs(f) > abs(i):
        if cases_sign[0]:
            return f-i # positive and pumping
        elif cases_sign[1]:
            return f+i # momentum turning negativ
        elif cases_sign[2]:
            return f+i # momentum turning positive
        else:
            return f-i # negative and dumping
    
    else:
        if cases_sign[0]:
            return f-i # positive and dumping
        elif cases_sign[1]:
            return f-i # momentum turning negativ
        elif cases_sign[2]:
            return -f-i # momentum turning positive
        else:
            return -f-i # negative but pumping (bottom is in ) 

def norm_augment(df,coin, mins = mins_dict, tops = tops_dict) :
    rng = range(len(df.index))
    top_price = tops[coin]
    botom_price = mins[coin]
    #first : store the price in the avg col 
    df.loc[:,'avg'] = df.loc[:,"close"].copy()#[(df["open"][i] + df["close"][i])/2 for i in rng]

    # !!! SUPER IMPORTANT price values are overwritten with their normalized values (min - max normalisation)
    # this step makes it possible to compare the MACD values across all coins 
    # MACD values are not calculated on the absolute prices but on their respective normalized values
    df.loc[:,"open"] = [(df["open"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"high"] = [(df["high"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"low"] = [(df["low"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"close"] = [(df["close"][i] - botom_price)/(top_price - botom_price) for i in rng]

    macd = bl.macd(df, pfast = 5, pslow = 16, psignal = 11)
    df = df.join(macd.df)
    df.dropna(inplace = True)
    #min_hist = df['histogram'].min()
    #max_hist = df['histogram'].max()
    df.loc[:,'histogram'] =[0 if abs(x) < 1e-06 else x for x in  df['histogram']]  # min - max normalisation on histogram 
    df.loc[:,'momentum'] = df.loc[:,'histogram'].rolling(2).apply(delta)
    #df.loc[:,'day_of_w'] = [weekDays[d.weekday()] for d in df.index]
    
    return df

record = {}
for c in coins:
    record[c] = pd.DataFrame( columns = ["sign", 'hyst'])


def mac_4D_strat (df):
    
    lbls = coins

    bought = False
    bags = {"fiat": 100}
    for c in lbls:
        bags[c]= 0

    for i in range(len(df.index)):
        ind = df.index[i]
        #ind2 = df.index[i+ 1] if not i > 2653  else df.index[i]
        #print(ind)
        #day = weekDays[ind.weekday()]

        histos = df.loc[ind, : ].filter(like = "histogram" )
        histos.sort_values(ascending=False, inplace= True)
    
        momentae = df.loc[ind, : ].filter(like = "momentum" )
        momentae.sort_values(ascending = False, inplace = True)
        # take snapshot of the amplitude and direction of the price movement 
        
        pair = histos.index[0].split("_")[0]
        pair_hist = histos[0]
        vector  = momentae.loc[f"{pair}_momentum"]

        if bought:
            #pair = snapshot["hist_max_idx"].split("_")[0]
            
        
            if bought in pair :
                if vector < 0 :
                    #strongest_momentum == pair:
                    #SELL
                    bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}BUSD_avg']
                    #print(f'2. Sold {pair}: bag is now {bags["fiat"]}$ ')


                    td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                    store = pd.DataFrame(td)
                    store.set_index("date", inplace= True)
                    record[bought] = pd.concat([record[bought], store], axis=0 )

                    bags[bought] = 0
                    bought = False
            
            else:
                bought_hist = histos.loc[f"{bought}BUSD_histogram"]
                if not (bought in  ["MINA", "AGIX"] or pair_hist > bought_hist * 1.09): # don't flip mina, agix or if histogram is more than 10% bigger 
                # convert "bought" using  BUSD
                    td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                    store = pd.DataFrame(td)
                    store.set_index("date", inplace= True)
                    record[bought] = pd.concat([record[bought], store], axis=0)


                    top_coin = [str for str in lbls if str in pair][0]
                    bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}BUSD_avg']
                    bags[top_coin] = bags['fiat'] / df.loc[ind, f'{top_coin}BUSD_avg']
                    bags["fiat"] = 0 
                    bags[bought] = 0
                    #print(f'3. Flipped {bought} for {top_coin}$ ')
                    bought = top_coin

                    td = {"date": [ind], "sign": [1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                    store = pd.DataFrame(td)
                    store.set_index("date", inplace= True)
                    record[bought] = pd.concat([record[bought], store], axis=0)


        else:
            
            #pair = snapshot["hist_max_idx"].split("_")[0]
            #vector  = momentae.loc[f"{pair}_momentum"]
            
            if vector > 0 :
                # BUY !
                #place buy order
                bought = [str for str in lbls if str in pair][0]
        
                bags[bought] = bags["fiat"] / df.loc[ind, f'{bought}BUSD_avg']
                bags["fiat"] = 0
                #print(f"4. Bought {bought}")

                td = {"date": [ind], "sign": [1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                store = pd.DataFrame(td)
                store.set_index("date", inplace= True)
                record[bought] = pd.concat([record[bought], store], axis=0)

    if bags["fiat"] == 0 :
        bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}BUSD_avg']
        bags[bought] = 0

    return bags["fiat"]



c_data = [import_coin_data(c) for c in data_files]
for i in range(len(coins)):
    c_data[i] = norm_augment(df = c_data[i], coin= coins[i])
    cols = c_data[i].columns
    c_data[i].columns = [f"{pairs[i]}_{col}" for col in cols]

test_subset = pd.concat(c_data[0:len(c_data) - 1], axis=1)
test_subset.dropna(inplace=True)

hists = test_subset.filter(like="histogram")
hists = hists.stack().reset_index()

vcts = test_subset.filter(like="momentum")
vcts = vcts.stack().reset_index()
    
print(hists.describe())



eval =  mac_4D_strat(test_subset)
print(f'Final evaluation at {eval } ')
