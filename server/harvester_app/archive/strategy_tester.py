#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import seaborn as sb
from sys import exit
from binance import Client
from datetime import datetime
import math
from matplotlib.dates import date2num



class ND_trading:
    def __init__(self, list_of_coins, price_extremes = None ) :
        self._coins = list_of_coins
        self._files = [f'./data/h/{coin}USDT.csv' for coin in self._coins]
        self._data = dict(zip (self._coins, [self._import_coin_data(c) for c in self._files]))
        self._cutoff = max([data.index[0] for data in self._data.values()])
        self._extremes = price_extremes or self._get_extremes()
        
        
        # For each coin add a entry in the record dict for potential plotting later also trim the data so all coin data start at the same time
        # Probably a bad idea to put these two unrelated things together !
        self._record = {} 
        for c in self._coins :
            self._record[c] = pd.DataFrame( columns = ["sign", 'hyst'])
            self._data[c] = self._trim(self._data[c])
            self._normalizer(c)
        # Put all data in one big data frame
        self._data = pd.concat(self._data.values(), axis= 1)
        # Record the market endpoints for comparison
        self._start_date = self._data.index[0]
        self._end_date = self._data.index[-1]
        self.start_date = self._start_date
        self.end_date = self._end_date

        self._start_prices = self._data.filter( like = "_price").iloc[0, : ]
        self._end_prices = self._data.filter( like = "_price").iloc[-1, : ]
        self._market_change = dict(
            zip( self._coins,
                [ ( (self._end_prices[f'{c}BUSD_price'] - self._start_prices[f'{c}BUSD_price']) /
                   self._start_prices[f'{c}BUSD_price'] ) *100 for c in self._coins]
                )
                )
        self._histograms = self._data.filter(like="histogram")
        self._momentae = self._data.filter(like="momentum")

        self._data["hist_total"] = self._histograms.sum(axis=1)
        self._data["hist_total_ma_8"] = self._data.loc[:,"hist_total"].rolling(8).apply(self._ma)
        self._market = sum(self._market_change.values()) / len(self._market_change.keys())
        self._x = np.array([date2num(d) for d in  self._data.index])
        self.x = self._x 
        
        self.record = self._record
        self._bought = False
        self.bought = self._bought
        self._bags = {"fiat": 100}
        for c in self._coins:
            self._bags[c]= 0
        
        self.bags = self._bags

        #First, second and third order derivatives of the MA8 of the mean of the normalized hisograms  
        self._data['dhdt_ma8'] = np.gradient(self._data["hist_total_ma_8"].values, self.x)
        self._data['d2hdt2_ma8'] = np.gradient(self._data['dhdt_ma8'].values, self.x)
        self._data['d3h_dt3_ma8'] = np.gradient(self._data['d2hdt2_ma8'].values, self.x)

        print(self._market_change)

    def _import_coin_data(self, pth):
        df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index, unit='ms')
        return df
    
    # Function that returns a dict with the max and min price values for each coin in the dataset
    def _get_extremes(self):
        extremes_dict = {}
        for c in coins:
            min_index = self._data[c].loc[self._cutoff:, "low"].idxmin()
            min_value = self._data[c].loc[min_index, "low"]
            
            max_index = self._data[c].loc[self._cutoff:, "high"].idxmax()
            max_value = self._data[c].loc[max_index, "high"]
            extremes_dict[c]= {
                #index and value
                "min": (min_index, min_value),
                "max": (max_index, max_value)
                } 
        return extremes_dict
    # Function that trims the loaded data to start at a specific date
    def _trim (self, coin_df):
        top = self._cutoff
        trimmed_df = coin_df.loc[top:] 
        return trimmed_df
    # Function that sets the normalized values for the open, high, low, close and also augments the data with macd hystogram and momentum
    def _normalizer(self, coin):
        top_price = self._extremes[coin]["max"][1]
        botom_price = self._extremes[coin]["min"][1]
        diff = top_price - botom_price
        rng = range(len(self._data[coin].index))
        self._data[coin]['price'] = self._data[coin]['close'].copy()

        self._data[coin]["open"] = [(self._data[coin]["open"][i] - botom_price)/diff for i in rng]
        self._data[coin]["high"] = [(self._data[coin]["high"][i] - botom_price)/diff for i in rng]
        self._data[coin]["low"] = [(self._data[coin]["low"][i] - botom_price)/diff for i in rng]
        self._data[coin]["close"] = [(self._data[coin]["close"][i] - botom_price)/diff for i in rng]

        macd = bl.macd(self._data[coin], pfast = 8, pslow = 63, psignal = 11)
        self._data[coin] = self._data[coin].join(macd.df)
        self._data[coin].loc[:,'histogram'] =[0 if abs(x) < 1e-03 else x for x in  self._data[coin]['histogram']] # reducing the noize 
        self._data[coin].loc[:,'momentum'] = self._data[coin].loc[:,'histogram'].rolling(2).apply(self._delta)
        cols = self._data[coin].columns
        self._data[coin].columns = [f"{coin}BUSD_{col}" for col in cols]
        self._data[coin].dropna(inplace = True)
    # Function that returns a momentum value. It is applyd to the macd histogram and retuns the diff between the two based on several cases  
    def _delta (self, arr):

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
    
    #function used to calculata a moving Average
    def _ma(self, arr):
        return sum(arr)/len(arr)
    
    # Function that applys the trading strat to each row in the dataset and returns a numerical value representing the final evaluation in % 
    def _mac_nD_strat (self):
        bought = False
        bags = {"fiat": 100}
        for c in self._coins:
            bags[c]= 0
    
        for i in range(len(self._data.index)):
            ind = self._data.index[i]

            histos = self._data.loc[ind, : ].filter(like = "histogram" )
            histos.sort_values(ascending=False, inplace= True)
        
            momentae = self._data.loc[ind, : ].filter(like = "momentum" )
            momentae.sort_values(ascending = False, inplace = True)
            # take snapshot of the amplitude and direction of the price movement 
            
            pair = histos.index[0].split("_")[0]
            pair_hist = histos[0]

            y = self._histograms[f"{pair}_histogram"].values
            self._data[f"{pair}_d1"] = np.gradient(y, self.x )
            self._data[f"{pair}_d2"] = np.gradient(self._data[f"{pair}_d1"].values, self.x )

            if bought:
                # get the coin price properties 
                y = self._histograms[f"{bought}BUSD_histogram"].values
                self._data[f"{bought}_d1"] = np.gradient(y, self.x )
                self._data[f"{bought}_d2"] = np.gradient(self._data[f"{bought}_d1"].values, self.x )
                # chec the global trend and SELL if DOWN
                #if self._data.loc[ind,'dhdt_ma8'] < 0 or  (abs(self._data.loc[ind,'dhdt_ma8']) < 0.01 and self._data.loc[ind,'d2hdt2_ma8'] < 0) :
                if self._data.loc[ind, f"{bought}_d1"] < 0 or  (abs(self._data.loc[ind,f"{bought}_d1"]) < 0.01 and self._data.loc[ind, f"{bought}_d2"] < 0) :
                    bags["fiat"] = bags[bought] * self._data.loc[ind, f'{bought}BUSD_price']
                    #print(f'2. Sold {pair}: bag is now {bags["fiat"]}$ '           
                    td = {"date": [ind], "sign": [-1], "hyst": [self._data.loc[ind, f'{bought}BUSD_histogram']]}
                    store = pd.DataFrame(td)
                    store.set_index("date", inplace= True)
                    self._record[bought] = pd.concat([self._record[bought], store], axis=0 )
                    bags[bought] = 0
                    bought = False


                elif f'{bought}BUSD' == pair :
                     pass
                    
                else:
                    bought_hist = histos[f'{bought}BUSD_histogram']
                    
                    if  pair_hist - bought_hist < 1e-06:
                            
                        td = {"date": [ind], "sign": [-1], "hyst": [self._data.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        self._record[bought] = pd.concat([self._record[bought], store], axis=0)

                        top_coin = [str for str in self._coins if f'{str}BUSD' == pair][0]
                        bags["fiat"] = bags[bought] * self._data.loc[ind, f'{bought}BUSD_price']
                        bags[top_coin] = bags['fiat'] / self._data.loc[ind, f'{top_coin}BUSD_price']
                        bags["fiat"] = 0 
                        bags[bought] = 0
                        #print(f'3. Flipped {bought} for {top_coin}$ ')
                        bought = top_coin

                        td = {"date": [ind], "sign": [1], "hyst": [self._data.loc[ind, f'{bought}BUSD_histogram']]}
                        store = pd.DataFrame(td)
                        store.set_index("date", inplace= True)
                        self._record[bought] = pd.concat([self._record [bought], store], axis=0)
                    
            else:
                if self._data.loc[ind, f"{pair}_d1"] > 0 or (abs(self._data.loc[ind,f"{pair}_d1"]) < 0.01 and self._data.loc[ind, f"{pair}_d2"] > 0) :

                    # BUY !
                    #place buy order
                    bought = [str for str in self._coins if f'{str}BUSD' == pair][0]
            
                    bags[bought] = bags["fiat"] / self._data.loc[ind, f'{bought}BUSD_price']
                    bags["fiat"] = 0
                    #print(f"4. Bought {bought}")

                    td = {"date": [ind], "sign": [1], "hyst": [self._data.loc[ind, f'{bought}BUSD_histogram']]}
                    store = pd.DataFrame(td)
                    store.set_index("date", inplace= True)
                    self._record[bought] = pd.concat([self._record[bought], store], axis=0)

        if bags["fiat"] == 0 :
            bags["fiat"] = bags[bought] * self._data.loc[ind, f'{bought}BUSD_price']
            bags[bought] = 0
        

        return bags["fiat"]-100
    
    def _plot_total_hist(self):
        #plt.plot( self._data["hist_total"] , linewidth = 0.7 , color = 'grey')
        plt.plot( self._data["hist_total_ma_8"] , linewidth = 0.7 , color = 'green')

        plt.axhline(y = 0, color = "red")
        plt.show()

    def _plot_hist_deriv(self):
        x = np.array([date2num(d) for d in  self._data.index])
        y = self._data["hist_total_ma_8"].values
        dy_dx = np.gradient(y, x)
        dy_dx = np.array([x  if abs(x) > 0.01 else 0 for x in dy_dx])
        zero_indices = np.where(dy_dx == 0)
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(self._data.index, y)
        ax[0].set_ylabel('y')
        ax[1].plot(self._data.index, dy_dx)
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('dy/dx')

        # Plot the zero points on the derivative plot
        ax[1].scatter(self._data.index[zero_indices], dy_dx[zero_indices], color='red', marker='o')

        plt.show()

    
    

    
    
#-------------------------------- | Choose coins to include in test here |--------------------------------------
coins = ['BTC', 'ADA', 'MINA', "DOT", "LINK"]
#----------------------------------------------------------------------------------------------------------------

#ND_trading(coins).print_cutoff()
dataset = ND_trading(coins)
print(f"Cosidered coins form {dataset.start_date} to {dataset._end_date}" )
for c in coins:
    print(c)
print(f'Average market performance is : {dataset._market} % :')
print(f'Tading strategy performance is : {dataset._mac_nD_strat()} % ')

#dataset._plot_total_hist()
#dataset._plot_hist_deriv()






    




    
""" 
def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

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

