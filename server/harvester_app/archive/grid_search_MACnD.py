#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import pickle


param_grid = {
    'pfast': [i for i in range(7,27)],
    'pslow' : [i for i in range (15, 69) ],
    'filter': [1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
}

search_grid = [
    (a,z,e)
    for a in param_grid['pfast']
    for z in param_grid['pslow']
    for e in param_grid['filter']
    if a < z-5 

]

of_interest =  ["ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"] #"COTI",0.05, 0.7,   "ERG"1.1 , 19.23


mins = [869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230, 1.1]
tops = [4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85, 19.23]

mins_dict = dict(zip(of_interest, mins))
tops_dict = dict(zip(of_interest, tops))

coins = ["PAXG", "AGIX", "KAVA", "FIL", "RNDR", "MATIC", "BTC", "ALGO", "MINA", "ADA"]
pairs = [f'{coin}BUSD' for coin in coins]
#data_files = [f'./data/h/{pair}_1h2.csv' if not pair == 'ERGBUSD' else "./data/h/ERGUSTD_1h.csv" for pair in pairs]
data_files =  [f'./data/h/{c}BUSD_1h2.csv'for c in coins]


mins_dict = dict(zip(of_interest, mins))
tops_dict = dict(zip(of_interest, tops))

def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def trim (coin_df):
    top = coin_df['high'].idxmax()
    trimmed_df = coin_df.loc[top: ] 
    return trimmed_df

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
            return i-f # positive and dumping
        elif cases_sign[1]:
            return f-i # momentum turning negativ
        elif cases_sign[2]:
            return -f-i # momentum turning positive
        else:
            return -f-i # negative but pumping (bottom is in )
    
#c_data = [pd.read_pickle(f"./data/h/pkls/{f'{c}BUSD'}.pkl") for c in of_interest]
def norm_augment(df,coin, pfast, pslow, filter, mins = mins_dict, tops = tops_dict) :
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

    macd = bl.macd(df, pfast = pfast, pslow = pslow, psignal = 11)
    df = df.join(macd.df)
    df.dropna(inplace = True)
    #min_hist = df['histogram'].min()
    #max_hist = df['histogram'].max()
    #df.loc[:,'histogram'] =( df['histogram']  - min_hist) / ( max_hist - min_hist) # min - max normalisation on histogram
    df.loc[:,'histogram'] =[0 if abs(x) < filter else x for x in  df['histogram']] # reducing the noize 
    df.loc[:,'momentum'] = df.loc[:,'histogram'].rolling(2).apply(delta)
    #df.loc[:,'day_of_w'] = [weekDays[d.weekday()] for d in df.index]
    return df

#========================================================================================

record = {}
for c in coins:
    record[c] = pd.DataFrame( columns = ["sign", 'hyst'])
c_data = [import_coin_data(c) for c in data_files]
#c_data = [pd.read_pickle(f"./data/h/pkls/{p}.pkl") for p in coins]

def mac_4D_strat (df):
    
    lbls = coins

    bought = False

    bags = {"fiat": 100}
    for c in coins:
        bags[c] = 0
   
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
            """ 
            else:
                bought_hist = histos.loc[f"{bought}BUSD_histogram"]
                if not bought in  ["MINA", "AGIX"] and pair_hist < bought_hist * 3:
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
            """
    

        else:
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
    
    print(bags["fiat"])
    
    return bags["fiat"]

def compute_gains(list_of_dfs, params ):
    augmented = []

    for i in range(len(list_of_dfs)):
        
        data = norm_augment(df=list_of_dfs[i], coin= of_interest[i], pfast= params[0], pslow=params[1], filter=params[2])
        cols = data.columns
        data.columns = [f"{pairs[i]}_{col}" for col in cols]
        augmented.append(data)
    
    test_subset = pd.concat(augmented, axis=1 )
    test_subset.dropna(inplace=True)
    gain = mac_4D_strat(test_subset)
    
    return gain, params


results = [compute_gains(c_data, params = p) for p in search_grid ]

with open("macnd_grid_resuts.pkl", "wb") as f:
    pickle.dump(results, f)

#print(f'Final evaluation at {eval } from {test_subset.index[0]} to {test_subset.index[-1]}')



""" 
# import data
btc = pd.read_csv(f'./data/btc_2017H_bars.csv', names=['date', 'open', 'high', 'low', 'close'])
btc.set_index('date', inplace=True)
btc.index = pd.to_datetime(btc.index, unit='ms')
top = btc['high'].idxmax()
# seleting the data from the top to present
btc =  btc.loc[top : ]

print(btc.head(5))

# trading optimisation function
def compute_gains_rsi_mcad (df, params ):
    rsi = bl.rsi(df, period = params[0])
    rsi_buy = params[1][1]
    rsi_sell = params[1][0]
    macd = bl.macd(df, pfast = params[2], pslow = params[3], psignal = params[4])
    df = df.join([rsi.df, macd.df])

    df = df.drop_duplicates(keep=False)
    df = df.dropna(axis=0)

    btc_bag = 1
    fiat_bag = 0
    buys = []
    sales = []

    for x in range( len(df.index) ):
    
        data = df.iloc[x][['close', 'rsi', 'macd', 'signal']]

        if data['macd'] > data["signal"] and data['rsi'] < rsi_buy and btc_bag != 0:
            sales.append(data['close'])
            fiat_bag = btc_bag * data['close'] #selling for fiat
            btc_bag = 0

        elif data['macd'] < data["signal"] and data['rsi'] > rsi_sell and btc_bag == 0:
            buys.append(data['close'])
            btc_bag = fiat_bag / data['close'] # buying btc
            fiat_bag = 0

        else:
            continue

    if btc_bag == 0: # if last operation was sell for fiat
        last_price = df.iloc[-1]["close"]
        buys.append(last_price) 
        btc_bag = fiat_bag /last_price # buy btc at the last price

    if len(buys) == 0 or len(sales) == 0: 
        gain = -99
    else:
        gain = btc_bag - 1
    

    return gain, params


results = [compute_gains_rsi_mcad(btc, params = p) for p in search_grid ]

scores = [r[0] for r in results] #  if isinstance(r, float)
best_score = max(scores)
print(best_score)
i = pd.Series(scores).idxmax()
best_combo = results[i][1]

print(f'Best performing parameteres are {best_combo}, with a gain of {best_score}% ')
pd.DataFrame(results).to_pickle('./data/results_m3_rsi_macd_grid_search_H.pkl')
"""