#!/usr/bin/python3

#import pickle
import pandas as pd 
import matplotlib.pyplot as plt
import btalib as bl
import datetime
from datetime import datetime

weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")


# First import data and process it
# takes csv data file from binance, adds indicators and returns a pandas df with dates as index
def import_coin_data(coin, period = "d"):
    df = pd.read_csv(f'./data/{coin}_2017{period}_bars.csv', names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    df['var'] = df["high"] - df["low"]
    df['var_prc'] = ((df["high"] - df["low"])/ (df['open']+ df['close'])/2)*100
    rsi = bl.rsi(df, period = 21)
    macd = bl.macd(df, pfast = 21, pslow = 63, psignal = 14)
    df = df.join([rsi.df, macd.df])
    return df

btc = import_coin_data("btc", period= 'H')
btc = btc.drop_duplicates(keep=False)
btc = btc.dropna(axis=0)
btc['good_h'] = btc["close"] - btc['open']
btc['magnitude'] = btc['good_h'].copy()
btc['good_h'][btc['good_h'] <= 0] = 0
btc['good_h'][btc['good_h'] > 0] = 1
btc['magnitude'][btc['magnitude'] <= 0] = (btc['high'] - btc['low']) * 100 / btc['high']
btc['magnitude'][btc['magnitude'] > 0] = (btc['high'] - btc['low']) * 100 / btc['low']
btc['day_of_w'] = [weekDays[d.weekday()] for d in btc.index]
btc['hour'] = [btc.index[h].time().hour for h in range(len(btc.index))]
print(len(btc.index)/168)

# HINT : Itterate over the 24 hours of the day
def lucky_h(h, df = btc):
    h_index = df['hour'] == h
    gd = df.loc[h_index]["good_h"].sum()
    cp = df.loc[h_index][["good_h" , "magnitude"]].copy()
    all= len(cp.index)
    cp["good_h"][cp['good_h'] == 0] = -1
    cp['tot'] =cp['good_h'] * cp['magnitude']
    return (gd, cp['tot'].sum(), all)

def volatile_hour(h, df = btc):
    h_index = df['hour'] == h
    vh = (btc.loc[h_index]["high"] - btc.loc[h_index]['low'])/((btc.loc[h_index]["open"]+btc.loc[h_index]["high"])/2)
    vhr = round(vh *100 , 2)
    return vhr.sum()


hs = len(btc['good_h'])
good_hs = btc['good_h'].sum()

#print(f'Avg P of up day since 2017: {good_hs/days}\nTotal days: {days}')
top = btc['high'].idxmax()

dst = len(btc['good_h'].loc[top : ])
gdst = btc['good_h'].loc[top : ].sum()
gdfst = gdst/dst

print(f'Avg P of up h: {round(good_hs/hs, 2)}\nAvg P of good h since top : {round(gdfst,2)}\n')

bdst_filter_index = btc['good_h'].loc[top : ] == 0.0
gdst_filter_index = btc['good_h'].loc[top : ] == 1.0

up_mag_st = btc.loc[top : ][gdst_filter_index]["magnitude"].sum()
down_mag_st = btc.loc[top : ][bdst_filter_index]["magnitude"].sum()
'''
#print(f'max UP {round(btc.loc[top : ][gdst_filter_index]["magnitude"].max(),2)}\nmax down: {round(btc.loc[top : ][bdst_filter_index]["magnitude"].max(),2)}')
#print(f'Up: {round(up_mag_st/dst , 2)} per day\nDown: {round(down_mag_st / dst, 2)} per day')

#print(f'It was a {weekDays[top.weekday()]}')

#print(btc.tail())
#print( weekDays[btc.index[0].weekday()] )
#print([weekDays[d.weekday()] for d in btc.index])


#print(btc.loc[top : ][gdst_filter_index].tail(15))
#print(btc.loc[top : ][bdst_filter_index].tail(15))

#ada = import_coin_data("ada")

# Any weekday more profitable than others ? 
print(btc.nunique( axis=0, dropna=True))
print(len(btc.index.unique()))
'''

for day in weekDays:
    x_day = btc.iloc[list(btc['day_of_w'] == day)]
    print("\n",day)
    for h in range(24) : 
        x, y, z = lucky_h(h=h, df = x_day)
        v = volatile_hour(h=h)
        print(f'{h} >> {x} >>> {round(y,2)}>>>{z} ---- > lycky h coef: {round(x/z , 2)} ----> v is {round(v/z , 2)}')
