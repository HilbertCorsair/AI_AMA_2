#!/usr/bin/python3
import pickle
import pandas as pd 
import matplotlib.pyplot as plt
# buy close tuesday sell close Wednesday buy open fryday and sell close saturday
with open('./data/btc_study.pkl', 'rb') as f:
    btc = pickle.load(f)
print(btc.head(5))
top = btc['high'].idxmax()
# seleting the data from the top to present
btc =  btc.loc[top : ]
fiat = 50000
btc_fund = 0
for x in range (len(btc.index)):
    print(round(fiat, 2))
    day = btc.iloc[x]['day_of_w']
    ope = btc.iloc[x]['open']
    cls = btc.iloc[x]['close']
    last_buy = 0
    last_sell = 0
    if fiat < 10:
        print(f'You broke bithc! You lose in {x} steps')
        break
    elif day == 'Tuesday':
        if cls < last_sell:
            btc_fund = fiat/cls # buy btc at closing price
            last_buy = cls
    elif day == 'Wednesday' :
        if btc_fund == 0:
            continue
        elif cls > last_buy:
            fiat = btc_fund * cls #sell btc at closing price
            last_sell = cls
    elif day == 'Friday':
        if ope < last_sell or last_sell == 0:
            btc_fund = fiat/ope # buy btc at open price
            last_buy = ope
    else:
        print('HODL !')

prc_down =  round(((1 - btc.iloc[-1]["low"] / btc.iloc[0]["high"]) *100) ,2) 

print(f'You lost {round((fiat/500) ,2 )} % in a market that lost {prc_down} %')
