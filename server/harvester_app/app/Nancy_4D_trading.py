import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import seaborn as sb
from sys import exit


weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")

data_files = [
    "./data/h/ADAUSDT_1h.csv",
    "./data/h/BTCUSDT_1h.csv",
    "./data/h/MINAUSDT_1h.csv",
    "./data/h/PAXGUSDT_1h.csv",
    "./data/h/ADABTC_1h.csv",
    "./data/h/MINABTC_1h.csv",
    "./data/h/PAXGBTC_1h.csv"
    ]
#["data/h/ADAUSDT_1h.csv",  "data/h/BTCUSDT_1df = last_338(df)h.csv",  "data/h/MINAUSDT_1h.csv",  "data/h/PAXGUSDT_1h.csv"]

pairs = ["ADAUSDT" ,"BTCUSDT", "MINAUSDT", "PAXGUSDT", "ADABTC", "MINABTC", "PAXGBTC"]

# First import data and process it
# takes csv data file from binance, adds indicators and returns a pandas df with dates as index

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
    s = arr[-1] - arr[0]
    return s 

def trim_norm_augment(df) :
    df = trim(df)
    rng = range(len(df.index))
    top_price = max(df.loc[:,'high'])
    botom_price = min(df.loc[:,"low"])
    #first : store the price in the avg col 
    df.loc[:,'avg'] = [(df["open"][i] + df["close"][i])/2 for i in rng]
    # !!! SUPER IMPORTANT price values are overwritten with their normalized values (min - max normalisation)
    # this step makes it possible to compare the MACD values across all coins 
    # MACD values are not calculated on the absolute prices but on their respective normalized values
    df.loc[:,"open"] = [(df["open"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"high"] = [(df["high"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"low"] = [(df["low"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"close"] = [(df["close"][i] - botom_price)/(top_price - botom_price) for i in rng]
    macd = bl.macd(df, pfast = 12, pslow = 26, psignal = 10)
    df = df.join(macd.df)

    df.loc[:,'momentum'] = df.loc[:,'histogram'].rolling(2, closed = "left").apply(delta)
    #df.loc[:,'day_of_w'] = [weekDays[d.weekday()] for d in df.index]
    return df
    
# df.iloc[:, 4:9] - > index , avg, macd, signal, histogram

#mina_macd = mina.iloc[:, 5:9]

c_data = [import_coin_data(c) for c in data_files]
c_data = [trim_norm_augment(df) for df in c_data]

coins = dict(zip(pairs, c_data))


for k, val in coins.items():
    cols = val.columns
    val.columns = [ f'{k}_{col}' for col in cols ]
    
"""

tail = 350
cutoff = len(coins["MINABTC"].index ) - tail
acutoff = len(coins["ADABTC"].index) - tail
bcutoff = len(coins["BTCUSDT"].index) - tail
ccutoff = len(coins["PAXGBTC"].index) - tail
dcutoff = len(coins["PAXGUSDT"].index) - tail

ada = coins["ADABTC"]["ADABTC_histogram"][acutoff:]
plt.plot(coins["MINABTC"]["MINABTC_histogram"][cutoff:], linewidth = 0.7 )
#plt.plot(coins["MINABTC"]["MINABTC_momentum"][cutoff:],color = "navy", linewidth = 0.7 )

plt.plot(ada, color = "olive",  linewidth = 0.7 )
plt.plot(coins["BTCUSDT"]["BTCUSDT_histogram"][bcutoff:], color = "red", linewidth = 0.7 )
#plt.plot(coins["BTCUSDT"]["BTCUSDT_momentum"][bcutoff:], color = "blue", linewidth = 0.7 )

plt.plot(coins["PAXGBTC"]["PAXGBTC_histogram"][ccutoff:], color = "orange", linewidth = 0.7 )
plt.plot(coins["PAXGUSDT"]["PAXGUSDT_histogram"][dcutoff:], color = "purple", linewidth = 0.7 )
#plt.plot(coins["PAXGUSDT"]["PAXGUSDT_momentum"][dcutoff:], color = "gold", linewidth = 0.7 )

plt.show()
"""
test_subset = coins["MINAUSDT"].iloc[38:,:]
test_subset = test_subset.join([coins["ADABTC"],coins["ADAUSDT"],coins["MINABTC"], coins["BTCUSDT"],coins["PAXGBTC"] ,coins["PAXGUSDT"]])

print(len(coins["MINAUSDT"].index))
print(len(test_subset.index))
del coins


def mac_4D_strat (df):
    histos = df.filter(like = "histogram" )
    momentae = df.filter(like = "momentum" )
    lbls = ["ADA", "MINA", "BTC", "PAXG", "AGIX"]


    bought = False

    bought_the_dipp = False
    

    actions = {
        "ADAUSDT_buy_price": [],
        "MINAUSDT_buy_price":[],
        "BTCUSDT_buy_price": [],
        "PAXGUSDT_buy_price": [],

        "ADAUSDT_sell_price": [],
        "MINAUSDT_sell_price":[],
        "BTCUSDT_sell_price": [],
        "PAXGUSDT_sell_price": [],


        "ADABTC_buy_price": [],
        "MINABTC_buy_price":[],
        "PAXGBTC_buy_price":[],

        "ADABTC_sell_price": [],
        "MINABTC_sell_price":[],
        "PAXGBTC_sell_price":[],

        "AGIXUSDT_sell_price":[],
        "AGIXBTC_sell_price":[],
        "AGIXBTC_buy_price":[],
        "AGIXUSDT_buy_price":[]
    }

    bags = {
        "fiat": 100, 
        "ADA": 0,
        "MINA": 0,
        "BTC": 0,
        "PAXG":0, 
        "AGIX": 0 
    }

    
    for i in range(len(df.index)):
        ind = df.index[i]
        print(ind)

        # take snapshot of the amplitude and direction of the price movement 

        snapshot = {
            "hist_min_idx" : histos.loc[ind, : ].idxmin(),
            "hist_min_val" : histos.loc[ind, : ].min(),
            "hist_max_idx" : histos.loc[ind, : ].idxmax(),
            "hist_max_val" : histos.loc[ind, : ].max(),
            "mom_min_idx" : momentae.loc[ind, :].idxmin(),
            "mom_min_val" : momentae.loc[ind, :].min(),
            "mom_max_idx" : momentae.loc[ind, :].idxmax(),
            "mom_max_val" : momentae.loc[ind, :].max()
        }

        if snapshot["hist_max_val"] > 0 and not bought :

            #chech momentum for the pair
            pair = snapshot["hist_max_idx"].split("_")[0]
            vector  = momentae.loc[ind, f"{pair}_momentum"]
            if vector > 0:
                # BUY !
                #place buy order
                bought = [str for str in lbls if str in pair][0]
        
                bags[bought] = bags["fiat"] / df.loc[ind, f'{bought}USDT_avg']
                print(bags[bought])

                bags["fiat"] = 0
                actions[f'{pair}_buy_price'].append(df.loc[ind, f'{pair}_avg'])
                print(f"1) Hei ! I just got in on  {bought}")

                for key, val in actions.items():
                    if not key == f'{pair}_buy_price':
                        val.append(np.nan)


        if snapshot["hist_max_val"] < 0 and not bought :

            pair = snapshot["hist_min_idx"].split("_")[0]
            vector =  momentae.loc[ind, ["MINAUSDT_momentum", "ADAUSDT_momentum", "BTCUSDT_momentum", "PAXGUSDT_momentum"]].max()
            strongest = momentae.loc[ind, ["MINAUSDT_momentum", "ADAUSDT_momentum", "BTCUSDT_momentum", "PAXGUSDT_momentum"]].idxmax()
            strongest = strongest.split("_")[0]
            
            # FIND OUT IF any vectors are positive and buy the strongest one 
            # if strongest vector is on the min hyst BUY THE DIPP 

            if vector > 0:
                bought = [str for str in lbls if str in strongest][0]
                bought_the_dipp = False if bought == "PAXG" or pair != strongest else True


                bags[bought] = bags["fiat"] / df.loc[ind, f'{bought}USDT_avg']
                bags["fiat"] = 0

                print(f"3) Hei ! I just bought some {bought}")

                actions[f'{pair}_buy_price'].append(df.loc[ind, f'{pair}_avg'])
                for key, val in actions.items():
                    if not key == f'{pair}_buy_price':
                        val.append(np.nan)
            else : #pass
                for key, val in actions.items():
                    val.append(np.nan)
        
        
        elif bought_the_dipp :
            # if bought is top pair and the momentum shifts sell to fiat (exit)
            pair = snapshot["hist_max_idx"].split("_")[0]
            vector  = momentae.loc[ind, f"{pair}_momentum"]
            if  bought in pair and vector < 0 :
                # TAKE PROFIT

                bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}USDT_avg']
                bags[bought] = 0

                print(f'5) This is MASIV !!! I bought the dipp and sold the topp for {bags["fiat"]}!!! ')
                print(f'Stash is {bags["fiat"]} on {ind}')

                action_pair = f'{bought}USDT'
                for key, val in actions.items():
                    if not key == action_pair:
                        val.append(np.nan)
                    else:
                        actions[f'{action_pair}_sell_price'].append(df.loc[ind, f'{pair}_avg'])
                        bought = False
                        print(f'Yo I just sold the top in {bought}')
                        print(f'Stash is {bags["fiat"]} on {ind}')
                
                bought = False
                bought_the_dipp = False

         #SWITCH AT CROSS ! 
        elif bought and not bought_the_dipp:
            # check top pair
            pair = snapshot["hist_max_idx"].split("_")[0]

            if not bought in pair:
                #check if BTC is on top 
                if pair == "BTCUSDT":
                    print(bought)
                    # SWITCH to BTC
                    #place COINBTC SELL ORDER OR COIN BUSD sell and BUSDBTC buy
                    sell_to_btc_pair = f'{bought}BTC'
                    actions[f'{sell_to_btc_pair}_sell_price'].append(df.loc[ind, f'{sell_to_btc_pair}_avg'])
                    print(f'6) Flipped {bought} for BTC !')

                    #Switch to BTC
                    bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}USDT_avg']
                    bags["BTC"] = bags["fiat"] / df.loc[ind, 'BTCUSDT_avg']
                    
                    bags["fiat"] = 0
                    print(bags)

                    bought = "BTC"
                    for key, val in actions.items():
                        if not key == sell_to_btc_pair:
                            val.append(np.nan)
                    

                elif bought == "BTC" and not pair == "BTCUSDT":
                    # Swithc from BTC to new top coin
                    top_coin = [str for str in ["ADA", "MINA", "PAXG"] if str in pair][0]
                    buy_to_btc_pair = f'{top_coin}BTC'
                    actions[f'{buy_to_btc_pair}_buy_price'].append(df.loc[ind, f'{buy_to_btc_pair}_avg'])
                    

                    bags["fiat"] = bags["BTC"] * df.loc[ind, 'BTCUSDT_avg']
                    bags[top_coin] = bags["fiat"] / df.loc[ind, f'{top_coin}USDT_avg']
                    bags["BTC"] = 0
                    bags["fiat"] = 0
                    bought = top_coin

                    print(f'7) Flipped BTC for {bought}')
                    print(bags)

                else:
                    print('8 STARTS HERE!')
                    print(bags["ADA"])
                    top_coin = [str for str in ["ADA", "MINA", "PAXG"] if str in pair][0]
                    print(top_coin)
                    sell_to_btc_pair = f'{bought}BTC'
                    buy_to_btc_pair = f'{top_coin}BTC'

                    # convert "bought" using BTC or BUSD
                    bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}USDT_avg']
                    print(bags["fiat"])
                    bags[top_coin] = bags['fiat'] / df.loc[ind, f'{top_coin}USDT_avg']
                    print(f"FIAT ptice for {top_coin}",  df.loc[ind, f'{top_coin}USDT_avg'])
                    print(bags[top_coin], bags["ADA"])
                    print(bought == "ADA")

                    bags["fiat"] = 0 
                    bags[bought] = 0
                    actions[f'{sell_to_btc_pair}_sell_price'].append(df.loc[ind, f'{sell_to_btc_pair}_avg'])
                    actions[f'{buy_to_btc_pair}_buy_price'].append(df.loc[ind, f'{buy_to_btc_pair}_avg'])

                    print(f'8) Converted {bought} to {top_coin}')
                    bought = top_coin

            # boght in pair but ...
            if  not  pair == "BTCUSDT" and bought == "BTC":
                # Swithc from BTC to new top coin

                top_coin = [str for str in ["ADA", "MINA", "PAXG"] if str in pair][0]
                buy_to_btc_pair = f'{top_coin}BTC'
                actions[f'{buy_to_btc_pair}_buy_price'].append(df.loc[ind, f'{buy_to_btc_pair}_avg'])
                    

                bags["fiat"] = bags["BTC"] * df.loc[ind, 'BTCUSDT_avg']
                bags[top_coin] = bags["fiat"] / df.loc[ind, f'{top_coin}USDT_avg']
                bags["BTC"] = 0
                bags["fiat"] = 0
                bought = top_coin

                print(f'10) Flipped BTC for {bought}')
                
            
            else:
                #Your're on top check momentum
                vector  = momentae.loc[ind, f"{pair}_momentum"]
                if vector < 0 :
                    # SELL the TOP ! 
                    bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}USDT_avg']
                    bags[bought] = 0
                    print(f'11) Just sold the topp for {bags["fiat"]} $ !!!')
                    print(f'Stash is {bags["fiat"]} on {ind}')
                    action_pair = f'{bought}USDT'
                    bought  = False

                    for key, val in actions.items():
                        if not key == action_pair:
                            val.append(np.nan)
                        else :
                            val.append(df.loc[ind, f'{action_pair}_sell_prince'])
                    
    
    print("====",bought, bags[bought], df.loc[ind, f'{bought}USDT_avg']) if bought else print(bought)
    if bags["fiat"] == 0 :
        bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}USDT_avg']
        bags[bought] = 0

    return bags["fiat"]
       

actions =  mac_4D_strat(test_subset)

print(actions)




''' 

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
v add extra dimensions to the trading strat
v check returns (backtest)
v fine tuneing
- code the trading bot
- run


else : # sell to fiat
                action_pair = f'{bought}USDT'
                for key, val in actions.items():
                    if not key == action_pair:
                        val.append(np.nan)
                    else:
                        print(f'2) Cahed out ! ')

                        bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}USDT_avg']
                        bags[bought] = 0

                        actions[f'{action_pair}_sell_price'].append(df.loc[ind, f'{pair}_avg'])
                        bought = False
                        print(f'2 Yo I just chashed out of {bought}')


elif snapshot["hist_max_val"] < 0 and not bought :  
            #chech momentum for the pair
            pair = snapshot["hist_min_idx"].split("_")[0]
            vector  = momentae.loc[ind, f"{pair}_momentum"]
            if vector > 0:
                # BUY THE DIPP !
                #place buy order
                bought = [str for str in ["ADA", "MINA", "BTC", "PAXG"] if str in pair][0]

                bags[bought] = bags["fiat"] / df.loc[ind, f'{bought}USDT_avg']
                bags["fiat"] = 0

                print(f"4) Hei ! I just bought the dipp in {bought}")
                actions[f'{pair}_buy_price'].append(df.loc[ind, f'{pair}_avg'])
                # variable that signals not to switch unless top and momentum changes
                bought_the_dipp = True
                for key, val in actions.items():
                    if not key == f'{pair}_buy_price':
                        val.append(np.nan)

            else :  # PASS 
                for key, val in actions.items():
                    val.append(np.nan)

'''

