#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import pickle


param_grid = {
    'rsi_period': [4 , 8, 12, 21, 24, 28, 42],
    'rsi_limits': [(65  , 45) , (66 , 44) , (68 , 42) , (70 , 30) , (75 , 25) , (78 , 22) , (80 , 20), (41 , 69)],
    'pfast': [5, 12, 21, 24],
    'pslow' : [26, 48 ,63, 72 ],
    'psignal': [10, 14, 21]
}

search_grid = [
    (a,z,e,r,t)
    for a in param_grid["rsi_period"]
    for z in param_grid["rsi_limits"]
    for e in param_grid['pfast']
    for r in param_grid['pslow']
    for t in param_grid['psignal']

]

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
