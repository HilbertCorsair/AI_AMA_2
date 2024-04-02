from pycoingecko import CoinGeckoAPI
import datetime as dt
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots

cg = CoinGeckoAPI()
of_interest = ['bitcoin', 'ethereum', "cardano", 'mina-protocol', "ergo"]

prices = cg.get_price(ids=['bitcoin', 'mina-protocol', 'ethereum',  "ergo", "cardano"], vs_currencies='eur')

def coin_converter (coin_a, coin_b, ref = "usd"):
    prices = cg.get_price(ids = [coin_a, coin_b], vs_currencies = ref)
    r = prices[coin_a][ref] / prices[coin_b][ref]
    return r
    
x = coin_converter('cardano', 'mina-protocol')
print(f'one ADA = {x} MINA therefore I have {(13000 * x)+2000} MINA')

for coin in of_interest :
    p= prices[coin]['eur']
    print(f'The price of {coin} is {p} euros')


print(prices)
print(prices['cardano'])
print(prices['cardano']['eur'])

def get_coin_data(coin):
    cache_path = './data/{}_data.pkl'.format(coin)
    try:
        f = open(cache_path, 'rb')
        data = pickle.load(f)
        print('Loaded {} from cache'.format(cache_path))
    except (OSError, IOError) as e:
        print('Downloading ...')
        data = pd.Series(cg.get_coin_market_chart_by_id(coin, vs_currency = 'usd', days = 700 )) 
        data.to_pickle(cache_path)
        print('Cached at {}'.format(cache_path))
    return data

all_data = [get_coin_data(c) for c in of_interest]   

#print(all_data[0:4])

btc_eth = pd.DataFrame(all_data[0]['market_caps'], columns = ['day', 'cap_btc']).join(
    other = pd.DataFrame(all_data[1]['market_caps'], columns = ['day', 'cap_eth']).set_index('day'), on  = "day")

ada_cap = pd.DataFrame(all_data[2]['market_caps'], columns = ['day', 'cap_ada'])
mina_cap = pd.DataFrame(all_data[3]['market_caps'], columns = ['day', 'cap_mina'])
erg_cap = pd.DataFrame(all_data[4]['market_caps'], columns = ['day', 'cap_erg'])
all_coins_cap = btc_eth.join(other= ada_cap.set_index('day'), on  = "day")


#btc_eth["day"]= pd.to_datetime(btc_eth['day'])

#btc_eth['total_cap_M'] = (btc_eth["cap_btc"] + btc_eth["cap_eth"]).map(lambda x: x * pow(10 , -6 ))# MILLION


#ada_cap['day'] = pd.to_datetime(ada_cap['day'], unit='ms', infer_datetime_format=True)
#all_coins_cap = btc_eth.join(other= ada_cap.set_index('day'), on  = "day")
all_coins_cap = all_coins_cap.join(other= mina_cap.set_index('day'), on  = "day")
all_coins_cap = all_coins_cap.join(other= erg_cap.set_index('day'), on  = "day")
all_coins_cap['day'] = pd.to_datetime(all_coins_cap['day'], unit="ms", infer_datetime_format=True)
#print(all_coins_cap.tail())


mina_cap['day'] = pd.to_datetime(mina_cap['day'], unit='ms', infer_datetime_format=True) 

erg_cap['day'] = pd.to_datetime(erg_cap['day'], unit='ms', infer_datetime_format=True) 





print(all_coins_cap.tail())


coins = cg.get_coins_list()
#[print (c['name'] + " >>> "  +c['id']) for c in coins]
mina_data = get_mina_data()
btc_eth = pd.DataFrame(mina_data['market_caps'], columns = ['mc', 'price'])
btc_eth['mc'] = pd.to_datetime(btc_eth['mc'], unit='ms', infer_datetime_format=True)

print(btc_eth)
#print(pd.DataFrame(mina_data['market_caps']))
#print(pd.to_datetime(btc_eth['mc'], unit='ms', infer_datetime_format=True))



ma_1 = 1 
ma_2 = 2
ma_3 = 3
ma_5 = 5
ma_8 = 8
ma_13 = 13

fib_series = [1,1,2,3]
while fib_series[-1] < 365 :
    fib_series.append(fib_series[-1] + fib_series[-2])
print(fib_series)



def add_ma (ma, data = btc_eth):   
    data[f'ma_{ma}'] = data['total_cap_M'].rolling(ma).mean()




mas_list = [ f'ma_{x}' for x in fib_series]


for x in [3,5,8,34,233]:
    add_ma(x)



plt.plot(btc_eth['total_cap_M'].iloc[500 : ], label = 'market cap', color = "black")  #it's actually the market cap !!!
plt.plot(btc_eth['ma_3'].iloc[500 : ], label = 'ma_3', color = "rosybrown")
plt.plot(btc_eth['ma_5'].iloc[500 : ], label = 'ma_5', color = "mediumorchid")
plt.plot(btc_eth['ma_8'].iloc[500 : ], label = 'ma_8', color = "darkviolet")
plt.plot(btc_eth['ma_34'].iloc[500 : ], label = 'ma_34', color = "navy")
plt.plot(btc_eth['ma_233'].iloc[500 : ], label = 'ma_233', color = "orange")

plt.legend(loc = 'upper left')
plt.show()
#print(btc_eth)



#for col in ["cap_btc", "cap_eth", "cap_ada", "cap_mina", "cap_erg"]:
#    all_coins_cap[f"{col}"] = np.log10(all_coins_cap[f"{col}"]) # transforming parket cap data to log 10




def add_rsi(df, col, days = 14, coin = 'btc'):
    delta_mcap = col.diff(1)
    delta_mcap.dropna(inplace = True)
    positive = delta_mcap.copy()
    negative = delta_mcap.copy()
    positive[positive < 0] = 0
    negative[negative > 0] = 0
    avg_gain = positive.rolling(window = days).mean()
    avg_loss = abs(negative.rolling(window = days).mean())
    r_strength = avg_gain / avg_loss
    print(r_strength)
    print(r_strength.shape)
    df[f'rsi_{days}_cap_{coin}'] = round(100.0 - (100.0/(1.0 + r_strength)))
    return(df)

for col in ["cap_btc", "cap_eth", "cap_ada", "cap_mina", "cap_erg"]:
    all_coins_cap[f"{col}"] = np.log10(all_coins_cap[f"{col}"]) # transforming parket cap data to log 10


all_coins_cap = add_rsi(all_coins_cap, all_coins_cap['cap_btc'])
all_coins_cap = add_rsi(all_coins_cap, all_coins_cap['cap_ada'] , coin = 'ada')

down_days = all_coins_cap['day'][all_coins_cap['rsi_14_cap_btc'] < 16] 
up_days = all_coins_cap['day'][all_coins_cap['rsi_14_cap_btc'] > 85] 

plt.figure(figsize= (12 , 8 ))
ax1 = plt.subplot(211)
ax1.plot(all_coins_cap['day'], all_coins_cap['cap_btc'])
ax1.set_axisbelow(True)
for d in down_days:
    ax1.axvline( d, linestyle = "--", color = "red", alpha = 0.5)

for u in up_days:
    ax1.axvline( u, linestyle = "--", color = "green", alpha = 0.5)


#ax1.axvline(down_days.all() , linestyle = '--', color = 'black')

ax2 = plt.subplot(212, sharex = ax1)
ax2.plot(all_coins_cap['day'], all_coins_cap['rsi_14_cap_btc'], color = "red")
ax2.plot(all_coins_cap['day'], all_coins_cap['rsi_14_cap_ada'], color = "black")



ax2.axhline(15 , linestyle = '--', color = 'purple')
ax2.axhline(30 , linestyle = '--', color = 'olive')
ax2.axhline(70 , linestyle = '--', color = 'pink')
ax2.axhline(85 , linestyle = '--', color = 'orange')


plt.show()




plt.plot(all_coins_cap["cap_btc"], label = "BTC", color = "black")
plt.plot(all_coins_cap['cap_eth'], label = "ETH", color = "navy" )
plt.plot(all_coins_cap['cap_ada'], label = "ADA", color = "blue" )
plt.plot(all_coins_cap['cap_mina'], label = "MINA", color = "mediumorchid")
plt.plot(all_coins_cap['cap_erg'], label = "ERG", color = "olive")


plt.legend(loc = 'upper left')
plt.show()


correlation_mat = all_coins_cap.drop(["day"], axis = 1 ).corr()

sns.heatmap(correlation_mat.iloc[:150], annot = True)
plt.show()


