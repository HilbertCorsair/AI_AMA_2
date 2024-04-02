import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import seaborn as sb
from sys import exit
from binance import Client
from datetime import datetime
import math


def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

#["data/h/ADAUSDT_1h.csv",  "data/h/BTCUSDT_1df = last_338(df)h.csv",  "data/h/MINAUSDT_1h.csv",  "data/h/PAXGUSDT_1h.csv"]
#pairs = [f'{coin}BUSD' for coin in of_interest]
coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ"#, "ERG"
    ]


mins = [
    15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
    869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230# , 1.1
    ]
tops = [
    69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
    4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85 #, 19.23
    ]

mins_dict = dict(zip(coins, mins))
tops_dict = dict(zip(coins, tops))

coins = [
    "BTC", "ADA", "MINA", "PAXG", #"AGIX",
    "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ"
    ]

#data_files = [f'./data/h/{pair}_1h2.csv' if not pair == 'ERGBUSD' else "./data/h/ERGUSTD_1h.csv" for pair in pairs]
#coins = [ "AGIX", "KAVA", "FIL", "RNDR", "MATIC", "BTC", "ALGO", "MINA"]
pairs = [f'{coin}BUSD' for coin in coins]
data_files = [f'./data/h/{pair}.csv' for pair in pairs ]
#data_files =  [f'./data/h/{c}BUSD_1h2.csv'for c in coins]

# First import data and process it
# takes csv data file from binance, adds indicators and returns a pandas df with dates as index
timestamp = 0
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
   

# this function can MAYBE be used to get the support /resistence levels 
# 2 params : arr size and the plus minus interval arround the initial value
def deriv (arr):
    xi = arr[0]
    xf = arr[1]
    limi = abs(xi) - abs(xi)*0.05
    lims = abs(xi) + abs(xi)*0.05
    diff = xf -xi

    deriv = diff if not (abs(diff) > limi and abs(diff) < lims ) else 0

    return deriv




#c_data = [pd.read_pickle(f"./data/h/pkls/{f'{c}BUSD'}.pkl") for c in of_interest]
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

    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    df = df.join(macd.df)
    df.dropna(inplace = True)
    #min_hist = df['histogram'].min()
    #max_hist = df['histogram'].max()
    df.loc[:,'histogram'] =[0 if abs(x) < 1e-03 else x for x in  df['histogram']] # reducing the noize 
    df.loc[:,'momentum'] = df.loc[:,'histogram'].rolling(2).apply(delta)
    #df.loc[:,'day_of_w'] = [weekDays[d.weekday()] for d in df.index]
    
    return df

#========================================================================================

c_data = [import_coin_data(c) for c in data_files]

#c_data = [pd.read_pickle(f"./data/h/pkls/{p}.pkl") for p in pairs]
   
for i in range(len(pairs)):
    c_data[i] = norm_augment(df = c_data[i], coin= coins[i])
    cols = c_data[i].columns
    c_data[i].columns = [f"{pairs[i]}_{col}" for col in cols]


  
# df.iloc[:, 4:9] - > index , avg, macd, signal, histogram

#mina_macd = mina.iloc[:, 5:9]

test_subset = pd.concat(c_data, axis=1 )

test_subset.dropna(inplace=True)

# ERG max 19.25, min 1.1
record = {}
for c in coins:
    record[c] = pd.DataFrame( columns = ["sign", 'hyst'])

def mac_4D_strat (df):
    
    lbls = coins #[ "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "SHIB", "LINK", "AR", "AAVE", "EGLD"]
    bought = False
    bags = {"fiat": 100}
    for c in lbls:
        bags[c]= 0

    #bags = {"BTC": 0 , "ADA": 0, "MINA": 0, "PAXG": 0, "AGIX": 0, "DOT": 0, "ALGO": 0, "BNB": 0, "MATIC": 0, "SHIB": 0, "LINK": 0, "AR": 0, "AAVE": 0, "EGLD": 0, "fiat": 100}
   
    for i in range(len(df.index)):
        ind = df.index[i]

        histos = df.loc[ind, : ].filter(like = "histogram" )
        histos.sort_values(ascending=False, inplace= True)
    
        momentae = df.loc[ind, : ].filter(like = "momentum" )
        momentae.sort_values(ascending = False, inplace = True)
        # take snapshot of the amplitude and direction of the price movement 
        
        pair = histos.index[0].split("_")[0]
        pair_hist = histos[0]
        vector  = momentae.loc[f"{pair}_momentum"]

        if bought:
            if f'{bought}BUSD' == pair :
                if vector < 0 : 
                    # flip for a positive vector if any
                    hope = (momentae > 0).any()
                    if hope :
                        positives = (momentae>0).items()
                        k = [k for k, v in positives if v ][0]
                        buy_pair = k.split("_")[0]
                        
                        td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        record[bought] = pd.concat([record[bought], store], axis=0)

                        top_coin = [str for str in lbls if f'{str}BUSD' == buy_pair][0]
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
                        
                        #strongest_momentum == pair:
                        #SELL
                        bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}BUSD_avg']
                        #print(f'2. Sold {pair}: bag is now {bags["fiat"]}$ '           
                        td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        record[bought] = pd.concat([record[bought], store], axis=0 )
                        bags[bought] = 0
                        bought = False
                
            else:
                bought_hist = histos[f'{bought}BUSD_histogram']
                bought_vector =momentae[f'{bought}BUSD_momentum']

                scenarios = {
                    "a": pair_hist < 0 and vector < 0 and bought_hist < 0 and bought_vector < 0,#- - - -   sell if no hope else flip
                    "b": pair_hist < 0 and vector > 0 and bought_hist < 0 and bought_vector > 0,#- + - +  ? hold if MINA or AGIX else flip
                    "c": pair_hist > 0 and vector > 0 and bought_hist < 0 and bought_vector < 0,#+ + - - flip
                    "d": pair_hist > 0 and vector > 0 and bought_hist < 0 and bought_vector > 0,#+ + - + hold ?
                    "e": pair_hist > 0 and vector > 0 and bought_hist > 0 and bought_vector < 0,#+ + + - flip
                    "f": pair_hist < 0 and vector < 0 and bought_hist < 0 and bought_vector > 0,#- - - + hold 
                    "g": pair_hist > 0 and vector > 0 and bought_hist > 0 and bought_vector > 0 # + + + +  > hold if MINA or AGIX else flip

                    }
                if scenarios["a"]:
                    hope = (momentae > 0).any()
                    if hope :
                        positives = (momentae>0).items()
                        k = [k for k, v in positives if v ][0]
                        buy_pair = k.split("_")[0]
                        
                        td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        record[bought] = pd.concat([record[bought], store], axis=0)

                        top_coin = [str for str in lbls if f'{str}BUSD' == buy_pair][0]
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
                        
                        #strongest_momentum == pair:
                        #SELL
                        bags["fiat"] = bags[bought] * df.loc[ind, f'{bought}BUSD_avg']
                        #print(f'2. Sold {pair}: bag is now {bags["fiat"]}$ '           
                        td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        record[bought] = pd.concat([record[bought], store], axis=0 )
                        bags[bought] = 0
                        bought = False

                elif scenarios["b"] or scenarios["g"] or scenarios["c"] or scenarios["d"] or scenarios["e"]:
                    bought_hist = histos.loc[f"{bought}BUSD_histogram"]
                    if not bought in  [ "AGIX", "MINA"] and pair_hist < bought_hist * 3:
                    # convert "bought" using  BUSD                
                        td = {"date": [ind], "sign": [-1], "hyst": [df.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        record[bought] = pd.concat([record[bought], store], axis=0)

                        top_coin = [str for str in lbls if f'{str}BUSD' == pair][0]
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
           
            if vector > 0 :
                # BUY !
                #place buy order
                bought = [str for str in lbls if f'{str}BUSD' == pair][0]
        
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



eval =  mac_4D_strat(test_subset)

print(f'Final evaluation at {eval } from {test_subset.index[0]} to {test_subset.index[-1]}')


"""
hists = test_subset.filter(like="histogram")
hists["total"] = hists.sum(axis=1)

# compute the FFT
hists_fft = np.fft.fft(hists["total"])

# compute the frequencies
freqs = np.fft.fftfreq(len(hists.iloc[:, 0]), 60*60)
print(hists.columns[0])

# shift the zero frequency component to the center of the spectrum
y_fft_shifted = np.fft.fftshift(hists_fft)
freqs_shifted = np.fft.fftshift(freqs)

df_frqs = pd.DataFrame({
    "frq": freqs_shifted,
    "ampl" : np.abs(y_fft_shifted)
})

mask = (df_frqs >= 0).all(axis=1)  # Create a mask selecting only rows with non-negative values
df_frqs = df_frqs[mask]

df_frqs = df_frqs.sort_values(by='ampl', ascending=False)
df_frqs["frq_weeks"] = df_frqs["frq"] * 60 *60 *24 *7*7*24

print(df_frqs.head(7))


# plot the spectrum
plt.plot(freqs_shifted, np.abs(y_fft_shifted))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0 , 0.000055)
plt.show()


plt.plot( hists["total"] , linewidth = 0.7 , color = 'blue')
plt.axhline(y = 0, color = "red")
plt.show()

tail = 96
cutoff = len(test_subset.index ) - tail
c_ind = test_subset.columns.get_loc('PAXGBUSD_histogram')
ag_ind = test_subset.columns.get_loc('AGIXBUSD_histogram')
paxg_hist = test_subset.iloc[cutoff :, c_ind]



buy_paxg = record["PAXG"][ record["PAXG"]["sign"] == 1 ]
sell_paxg = record["PAXG"][ record["PAXG"]["sign"] == -1 ]
x_buy_paxg = [i for i in buy_paxg.index if i in paxg_hist.index]
x_sell_paxg = [i for i in sell_paxg.index if i in paxg_hist.index]

AGIX_hist = test_subset.iloc[cutoff :, ag_ind]
buy_AGIX = record["AGIX"][ record["AGIX"]["sign"] == 1 ]
sell_AGIX = record["AGIX"][ record["AGIX"]["sign"] == -1 ]
x_buy_AGIX = [i for i in buy_AGIX.index if i in AGIX_hist.index]
x_sell_AGIX = [i for i in sell_AGIX.index if i in AGIX_hist.index]

buy_both = [i for i in sell_AGIX.index if i in x_buy_paxg]
sell_both = [i for i in buy_AGIX.index if i in x_sell_paxg]



plt.plot( paxg_hist , linewidth = 0.7 , color = 'gold')
plt.plot( AGIX_hist , linewidth = 0.7 , color = 'blue')

#plt.scatter(x= x_sell_paxg, y= record["PAXG"].loc[x_sell_paxg,'hyst'],marker  ="v", color ="red")
#plt.scatter(x= x_buy_paxg, y= record["PAXG"].loc[x_buy_paxg,'hyst'],marker  ="^", color ="green")

#plt.scatter(x= x_sell_AGIX, y= record["AGIX"].loc[x_sell_AGIX,'hyst'],marker  ="v", color ="red")
#

plt.scatter( x= buy_both, y= record["AGIX"].loc[buy_both,'hyst'], marker  ="v" , color ="red" )
plt.scatter( x= sell_both, y= record["AGIX"].loc[sell_both,'hyst'], marker  ="^" , color ="green" )

plt.scatter( x= buy_both, y= record["PAXG"].loc[buy_both,'hyst'], marker  ="^" , color ="green" )
plt.scatter( x= sell_both, y= record["PAXG"].loc[sell_both,'hyst'], marker  ="v" , color ="red" )

plt.show()


tail = 96
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

#test_subset = coins["MINAUSDT"].iloc[38:,:]
#test_subset = test_subset.join([coins["ADAUSDT"], coins["BTCUSDT"],coins["PAXGUSDT"],coins["LINKUSDT"],coins["ARUSDT"],coins["SHIBUSDT"]])

test_subset = pd.concat(coins.values(), axis=1 )
print(test_subset.shape())
print(test_subset.tail())
print(test_subset.columns)
exit()
del coins
"""

