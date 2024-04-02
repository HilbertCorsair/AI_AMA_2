import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import seaborn as sb
import math
from sys import exit
weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")

data_files = ["data/h/ADAUSDT_1h.csv",  "data/h/BTCUSDT_1h.csv",  "data/h/MINAUSDT_1h.csv",  "data/h/PAXGUSDT_1h.csv"]

#[ "./data/h/BTCUSDT_1h.csv",  "./data/h/MINAUSDT_1h.csv", "./data/h/PAXGUSDT_1h.csv",
#"./data/h/ADABTC_1h.csv", "./data/h/MINABTC_1h.csv",  "./data/h/PAXGBTC_1h.csv"  ]
# First import data and process it
# takes csv data file from binance, adds indicators and returns a pandas df with dates as index




def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def trim (coin_df):
    top = coin_df['high'].idxmax()
    trimmed_df = coin_df.loc[top : , ] 
    return trimmed_df

def sum_o_square (window):
    s = sum([x**2 for x in window])
    return s 

def trim_norm_augment(df) :
    df = trim(df)
    top_price = max(df['high'])
    botom_price = min(df["low"])

    df.loc[:,'avg'] = round((df["open"] + df["close"])/2, 3)
    df.loc[:,"open"] = (df["open"] - botom_price)/(top_price - botom_price)
    df.loc[:,"high"] = (df["high"] - botom_price)/(top_price - botom_price)
    df.loc[:,"low"] = (df["low"] - botom_price)/(top_price - botom_price)
    df.loc[:,"close"] = (df["close"] - botom_price)/(top_price - botom_price)
    
    macd = bl.macd(df, pfast = 12, pslow = 26, psignal = 10)
    df = df.join(macd.df)
    df.loc[:,'momentum'] = df.loc[:,'histogram'].rolling(12, closed = "left").apply(sum_o_square)
    #df.loc[:,'day_of_w'] = [weekDays[d.weekday()] for d in df.index]
    return df



# replace prices by df["avg"], and macd_data by df["col_name"]
def mac_1D_strat (df):
    qtile = df["momentum"].quantile(0.98)
    buy_price = []
    sell_price = []
    macd_signal = []
    bought = False
    for i in range(len(df.index)):
        up = df['macd'][i] > df['signal'][i]
        # buy conditions
        if up and not bought :# and df['day_of_w'][i] in ["Tuesday", "Fryday", "Saturday"] :
            bought = True
            buy_price.append(df["avg"][i])
            sell_price.append(np.nan)
            macd_signal.append(1)
        
        # Sell conditions
        elif not up and bought or (df["momentum"][i] > qtile and bought):
            buy_price.append(np.nan)
            sell_price.append(df["avg"][i])
            bought = False
            macd_signal.append(-1)
        
        # Pass
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
    
    df.loc[:,"buy_price"] = buy_price
    df.loc[:,"sell_price"] = sell_price
    df.loc[:,'macd_signal'] = macd_signal
    return df

# df.iloc[:, 4:9] - > index , avg, macd, signal, histogram

#mina_macd = mina.iloc[:, 5:9]

ada = import_coin_data("./data/h/ADAUSDT_1h.csv")
ada = trim_norm_augment (ada)
ada =  mac_1D_strat(ada)

btc = import_coin_data(data_files[1])
btc = trim_norm_augment (btc)
btc =  mac_1D_strat(btc)

mina = import_coin_data(data_files[2])
mina = trim_norm_augment (mina)
mina =  mac_1D_strat(mina)

paxg = import_coin_data(data_files[3])
paxg = trim_norm_augment (paxg)
paxg =  mac_1D_strat(paxg)

coins = {
    "ada": ada,
    "btc": btc,
    "mina": mina,
    "paxg": paxg
}


for key, val in coins.items():

    # change col names
    cols = val.columns
    val.columns = [f'{key}_{name}' for name in cols]

    tmp_slice = val[[f"{key}_buy_price", f"{key}_sell_price"]]
    bag_fiat = 100
    bag_crypto = 0
    gains = 0
    
    
    for i in range (len(tmp_slice.index)):

        # buy conditions
        if not math.isnan(tmp_slice[f"{key}_buy_price"][i]) :
            bag_crypto = bag_fiat / tmp_slice[f"{key}_buy_price"][i]
            #print(f'bought {bag_crypto} {key} for {bag_fiat} $')
            bag_fiat = 0

        # sell conditions
        elif not math.isnan(tmp_slice[f"{key}_sell_price"][i]):
            bag_fiat = bag_crypto * tmp_slice[f"{key}_sell_price"][i]
            #print(f'sold {bag_crypto} {key} for {bag_fiat} $')
            bag_crypto = 0
        else:
            pass
    
    if bag_crypto > 0:
        gains = bag_crypto * val[f"{key}_avg"][-1]
    else:
        gains = bag_fiat
  
    margin = val.loc[:,f"{key}_sell_price"].sum() - val.loc[:,f"{key}_buy_price"].sum()
    print(f'\n\n\n{key} --- {round(margin,2)} with > {round(gains)} % gains\n{len(val.index)} rows')

coins = ada.join(other = [btc, mina , paxg])


del ada
del btc
del mina
del paxg

#tmp_df = coins.filter(like='histogram')
#print(tmp_df.iloc[784, :].idxmax())
#print(tmp_df.columns)
#print(coins.tail(3))
#print(coins.iloc[30:50, :])



up = { 
        "ada": coins['ada_macd'][0] > coins['ada_signal'][0],
        "mina": coins['mina_macd'][0] > coins['mina_signal'][0],
        "btc": coins['mina_macd'][52] > coins['mina_signal'][52],
        "gdax": coins['mina_macd'][300] > coins['mina_signal'][300]
        }

print(f'{up["ada"]}\n{up["ada"] == True}\n{up["ada"] == np.nan}')
"""
def mac_4D_strat (df): # buy the min and sell the max
    tmp_df = coins.filter(like='histogram')
    cols = tmp_df.columns
    lbls = [cn.split("_")[0] for cn in tmp_df.columns]
    bought = False

    #qtile = df["momentum"].quantile(0.01)
    prices = {
        "ada_buy" : [],
        "ada_sell" : [],
        "btc_buy" : [],
        "btc_sell" : [],
        "mina_buy" : [],
        "mina_sell" : [],
        "paxg_buy" : [],
        "paxg_sell" : []
    }
    

    #macd_signal = []
    
    for i in range(len(tmp_df.index)):
        promo = tmp_df.iloc[i, :].min()
        
        
        
        match promo :
            case  np.nan: 
                for k, v in prices.items :
                    v.append[np.nan]

            
            case 'ada_histogram':
                ada_buy.append[df["avg"][i]]
                ada_sell.append[np.nan]

                btc_buy.append[np.nan]
                btc_sell.append[np.nan]

                mina_buy.append[np.nan]
                mina_sell.append[np.nan]

                paxg_buy.append[np.nan]
                paxg_sell.append[np.nan]




        up = df['macd'][i] > df['signal'][i]
        # buy conditions
        if up and not bought:# and df['day_of_w'][i] in ["Tuesday", "Fryday", "Saturday"] :
            bought = True
            buy_price.append(df["avg"][i])
            sell_price.append(np.nan)
            macd_signal.append(1)
        
        # Sell conditions
        elif not up and bought:# and df["momentum"][i] > qtile:
            buy_price.append(np.nan)
            sell_price.append(df["avg"][i])
            bought = False
            macd_signal.append(-1)
        
        # Pass
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
    
    df.loc[:,"buy_price"] = buy_price
    df.loc[:,"sell_price"] = sell_price
    df.loc[:,'macd_signal'] = macd_signal
"""
    #return df




''' 
qtile = mina["momentum"].quantile(0.7)
plt.figure(figsize = (9,9))
sb.kdeplot(mina['momentum'] , fill = True)
plt.axvline(0 , color = 'orange')
plt.axvline(qtile , linestyle = '--', color = 'purple')
plt.show()


def plot_macd(df): #(df)
    prices = df["avg"]
    macd = df["macd"]
    signal = df["signal"]
    hist = df["histogram"]
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(prices)
    ax1.plot(df.index, df["buy_price"], marker = '^', color = 'green', markersize = 5, label = 'BUY', linewidth = 0)
    ax1.plot(df.index, df["sell_price"], marker = 'v', color = 'red', markersize = 5, label = 'SELL', linewidth = 0)
    ax1.legend()
    ax1.set_title('1D MACD trading strat')
    ax2.plot(macd, color = 'olive', linewidth = 0.7, label = 'MAC-D')
    ax2.plot(signal, color = 'navy', linewidth = 0.7, label = 'signal')

    for i in range(len(df.index)):
        if hist[i] < 0:
            ax2.bar(prices.index[i], hist[i], width= 0.03, color = '#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], width= 0.03, color = '#26a69a')

    plt.legend(loc = 'lower right')
    plt.show()

plot_macd(mina)
'''
'''
LEFT TO DO:
v test model as is to get a baseline for returns
v reduce noize by looking at the past 5 - 10 hyst and do the stats on the sum of the hist squares 
v test new model and compare returns ----> momentum on past macd hystgrams doesn't help
v improove the signal by looking at the price evolution since last trade - stats on the momentum to determine the ideal sell / buy point
v test new model and compare returns 
v grid search the parametes sapce for the best model: params : romming window size for momentum, percentile of momentum
- add extra dimensions to the trading strat
- check returns (backtest)
- code the trading bot
- run
'''

