#!/usr/bin/python3
import pickle
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
import btalib as bl
import math
from sys import exit
weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")


# First import data and process it
# takes csv data file from binance, adds indicators and returns a pandas df with dates as index
def import_coin_data(coin, period = "d"):
    df = pd.read_csv(f'./data/{coin}_2017{period}_bars.csv', names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    df['12sma'] = df.close.rolling(12).mean()
    df['26sma'] = df.close.rolling(26).mean()
    df['var'] = df["high"] - df["low"]
    df['var_prc'] = ((df["high"] - df["low"])/ (df['open']+ df['close'])/2)*100
    rsi = bl.rsi(df, period = 4)
    macd = bl.macd(df, pfast = 12, pslow = 26, psignal = 10)
    df = df.join([rsi.df, macd.df])
    return df

def import_coin_data_new(path = './data/btc_H_updated.pkl'):
    with open(path, 'rb') as f:
        df = pickle.load(f)

    rsi = bl.rsi(df, period = 4)
    macd = bl.macd(df, pfast = 12, pslow = 26, psignal = 10)
    df = df.join([rsi.df, macd.df])
    return df


#btc = import_coin_data_new()
btc = import_coin_data("btc", period='H')
#exit(0)
top = btc['high'].idxmax()
# seleting the data from the top to present
#btc =  btc.iloc[44054 : ]
btc = btc.loc[top : ]
btc['day_of_w'] = [weekDays[d.weekday()] for d in btc.index]
btc['hour'] = [btc.index[h].time().hour for h in range(len(btc.index))]


btc = btc.drop_duplicates(keep=False)
btc = btc.dropna(axis=0)


buy = []
sell = []

btc_bag = 1 # start wit 1 BTC
fiat_bag = 0

print(btc.head(5))
exit()
for x in range( len(btc.index) ):
    
    data = btc.iloc[x][['close', 'rsi', 'macd', 'signal']]
    t_minus_1 = btc.iloc[x-1][['high', 'low']]
    t_minus_2 = btc.iloc[x-2][['high', 'low']]


    if data['macd'] > data["signal"] and data['rsi'] < 69 and btc_bag != 0: #  sell BTC if you have it
        fiat_bag = btc_bag * data['close']
        btc_bag = 0
        sell.append(data['close'])
        buy.append(float('nan'))
       
    elif data['macd'] < data["signal"] and data['rsi'] > 41  and btc_bag == 0: #  buy BTC
        btc_bag = fiat_bag / data['close']
        fiat_bag = 0
        sell.append(float('nan'))
        buy.append(data['close'])
       
    else :  
        buy.append(float('nan'))
        sell.append(float('nan'))

if fiat_bag == 0:
    print(f'a) Gain is {btc_bag -1} BTC ')
else:

    sales = [x for x in sell if math.isnan(x) == False]

    sf =  btc.iloc[-1]['close'] # sale at last price 
    s1 = sales[0] # first sale price (ref)
    print(f"b) Gain is {round(((fiat_bag / sf)-1 )*100 , 2)}% BTC")


btc["buy"] = buy
btc['sell'] = sell
btc = btc.iloc[(9522 -1500) : ]

plt.figure(figsize= (12 , 8 ))
ax1 = plt.subplot(211)
ax1.plot(btc['close'], label = 'BTC price', alpha = 0.5, color = "blue")
ax1.scatter(btc.index, btc["buy"], marker  ="^", color = "green")
ax1.scatter(btc.index, btc["sell"], marker  ="v", color = "red")
ax1.set_axisbelow(True)

ax2 = plt.subplot(212, sharex = ax1)
ax2.plot(btc.index, btc['rsi'],color = "black")



ax2.axhline(69 , linestyle = '--', color = 'purple')
ax2.axhline(41 , linestyle = '--', color = 'olive')


plt.show()



   











