#!/usr/bin/python3

from webbrowser import get
import pandas as pd 
import btalib as bl
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from binance import Client
from csv import writer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy import stats
#C6e7yMMgkArr3529

weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
# ! reads API credentials from file and returns a connection object to the binance account
def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

def get_latest_time_stamp(pair ):
    df = pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{pair}.pkl")
    return df['date'].iloc[-1]

# takes csv data file from binance, adds indicators and returns a pandas df with dates as index
def import_coin_data (pair):
    df = pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{pair}.pkl")
    #df.set_index('date', inplace=True)
    #df.index = pd.to_datetime(df.index, unit='ms')
    return df

def by_week_day_stats(pair):
    df = import_coin_data(pair).tail(365*24)#last 1 year
    rng = range(len(df.index))
    df.loc[:, 'delta'] = [(df["close"][i] - df["open"][i]) for i in rng]
    m = btc["delta"].mean()
    st_dev = df["delta"].std()
    interval = (m - st_dev, m, m + st_dev)

    by_week_day = df.groupby("day_of_w").agg({
       'delta':'sum',
        'delta':'mean'
    })

    print(by_week_day)
    print(interval)

btc = import_coin_data('BTCUSDT')

n_obs = len(btc.index)
btc['day_of_w'] = [weekDays[d.weekday()] for d in btc.index]
btc["hour"] = [d.hour for d in btc.index]
rng = range(len(btc.index))
btc.loc[:,'avg'] = [(btc["open"][i] + btc["close"][i])/2 for i in rng]
btc.loc[:,'move'] = [0 if x > 0 else 1 for x in btc['close'] - btc['open']]
btc.loc[:, 'delta'] = [(btc["close"][i] - btc["open"][i]) for i in rng]

btc = btc.tail(900*24)


m = btc["delta"].mean()
st_dev = btc["delta"].std()
interval = (m - st_dev, m, m + st_dev)
variance = ((2*st_dev/m) * 100)


btc_by_week_day = btc.groupby("day_of_w").agg({
    'delta':'sum',
    'delta':'mean'
})


# Group by 'Weekday' and calculate mean sales for each day
weekdays = btc.groupby('day_of_w')['delta'].mean().reset_index()
print(weekdays)

# Function to perform t-test against overall mean
def perform_ttest(group):
    t_stat, p_val = stats.ttest_1samp(group, 2.46)
    return pd.Series({'t-statistic': t_stat, 'p-value': p_val})

# Group by 'Weekday', apply the t-test function on each group's sales
ttest_results = btc.groupby('day_of_w')['delta'].apply(perform_ttest).reset_index()
print(ttest_results)



print(btc_by_week_day)
print(interval)


def test_strat(pair):
    df = import_coin_data(pair).tail(24*365)
    df['day_of_w'] = [weekDays[d.weekday()] for d in df.index]
    df["hour"] = [d.hour for d in df.index]
    rng = range(len(df.index))
    df.loc[:,'avg'] = [(df["open"][i] + df["close"][i])/2 for i in rng]
    df.loc[:,'move'] = [0 if x > 0 else 1 for x in df['close'] - df['open']]
    df.loc[:, 'delta'] = [(df["close"][i] - df["open"][i]) for i in rng]



    buy = []
    sell = []
    fiat_bag = 100
    btc_bag = 0

    for i in rng:

        if  df["day_of_w"][i] != "Thursday" and  btc_bag == 0:
            sell.append(float('nan'))
            buy.append(df['close'][i]) # <---- buying price
            btc_bag = fiat_bag / df["close"][i]
            fiat_bag = 0
            

        elif df["day_of_w"][i] == "Thursday" and fiat_bag == 0:
            sell.append(df["close"][i]) # <----------- selling price
            buy.append(float('nan'))
            fiat_bag = btc_bag * df["close"][i]
            btc_bag = 0
            

        else : 
            buy.append(float('nan'))
            sell.append(float('nan'))
    
    final_evaluation = fiat_bag if fiat_bag !=0 else btc_bag * df["close"][-1]
    final_evaluation = round(final_evaluation -100 , 2)

    market  = round(((df["close"][-1] -  df["close"][0])/df["close"][0] ) * 100, 2)

    print(f"Congratulations your strategy made {final_evaluation}%")
    print(f"Meanwhile the market made {market}")


print(test_strat("BTCUSDT"))

''' 

min_price = min(btc["low"]) 
max_price = max(btc["high"]) 
print(min_price, max_price)

btc_by_wh = btc.iloc[n_obs - 27000 :, :]

btc_by_wh = btc_by_wh.groupby(["day_of_w", "hour"])


summary_btc_wh = btc_by_wh['move'].agg(['mean', 'std', 'count'])
#summary_btc_momentum = btc_by_wh['momentum'].agg(['mean', 'std'])


def getp (line):
    p_hat = round(line[0] * line[2])
    n_obs = int(line[2])
    phy = 0.5
    pval = binomtest(p_hat, n_obs, phy)
    return pval.pvalue
summary_btc_wh["p_val"] = [getp(line) for line in summary_btc_wh.values]

summary_btc_wh = summary_btc_wh.sort_values('p_val', ascending = True)


day_of_w hour                                     
Friday   19    0.385093  0.488136    161  0.004403
Tuesday  14    0.600000  0.491436    160  0.014001
Friday   2     0.597484  0.491954    159  0.017077
Thursday 3     0.596273  0.492175    161  0.017787
Sunday   15    0.403727  0.492175    161  0.017787
Thursday 10    0.590062  0.493356    161  0.027024
'''



'''
def update_btc_candels(file_name = './data/btc_2017d_bars.csv'):
    ts = int(get_latest_time_stamp())
    cli = unlock()
    new_data = cli.get_historical_klines('BTCUSDT', '1d', ts, limit=1000)
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        for line in new_data:
            csv_writer.writerow(f'{line[0]}, {line[1]}, {line[2]}, {line[3]}, {line[4]}\n')
    write_obj.close()
    print('Updated !')
    



#btc = pd.read_pickle("/home/honeybadger/projects/harvester/data/h/pkls/BTCBUSD.pkl")

# INDICATORS
ma2 = bl.SMA(btc['close'], period= 15)
#macd = bl.macd(btc, pfast = 8, pslow = 63, psignal = 11)
#rsi = bl.rsi(btc, period = 8)
btc = btc.join([ ma2.df])
btc.loc[:,'sma'] = btc.loc[:,'sma'].rolling(window=9).mean()
btc.dropna(inplace = True)
#--------

# DERIVATIVE 1 
x = np.array([date2num(d) for d in  btc.index])


y_sma = btc.loc[:,'sma'].values
#y_hist = btc.loc[:, "histogram"]
btc.loc[:, "d1_sma_8"] = np.gradient(y_sma, x)
#btc.loc[:, "d1_hyst_macd_8_63"] = np.gradient(y_hist, x)
#---------------------------------------------------------------------


l300btc = btc.loc[ btc.index[-5000] : , : ]
prices = l300btc[ abs(l300btc["d1_sma_8"]) <= 15]
prices_lm = prices[abs(l300btc["close"] - 27700)<= 500]
#print(prices_lm.loc[:,"close"])
print(prices.shape, l300btc.shape)

plt.plot(prices["close"], "o")
plt.plot(l300btc["sma"])
plt.plot(l300btc["close"], color = "gold")

ax = plt.gca()

ax.yaxis.set_major_locator(ticker.MultipleLocator(base=100))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(base=50))
plt.show()

X = pd.to_numeric(prices_lm.index)
y = prices_lm['close'].values

points = prices_lm.loc[: , "close"]

def gradient_descent (mi, bi, points,  L):
    m_gradient = 0
    b_gradient = 0
    n = len(prices_lm.index)
    for i in range(n):
        x = pd.to_numeric(points.index)[i] / 1000
        
       
        y = points.iloc[i]
       
        m_gradient += -(2/n) * x * (y - (x * mi + bi))
        b_gradient += -(2/n) * (y - (x * mi + bi))
        
       
    
    m = mi - m_gradient * L
    b = bi - b_gradient * L
    

    return m , b 

m = 0
b = 0 
L = 0.0001
epochs = 10

for i in range(epochs):
    print(i)
    
    m, b = gradient_descent(mi=m, bi=b, points= points, L = L)
    print(m, b)
    




plt.scatter([x.timestamp() for x in points.index], points, color = "blue")
plt.plot( list(range(min(X), max(X)), [m *x + b for x in range( min(X), max(X) )] ))

#
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema

price_lvls = btc[(abs(btc["d1_sma_8"]) <= 15) & (btc['avg'] > 25000)]
price_lvls = [round(x) for x in price_lvls["avg"].values]

density = gaussian_kde(price_lvls)
xs = np.linspace(25000, 64800, 35000)
density.covariance_factor = lambda: .03
density._compute_covariance()

# Compute relative extrema (peaks)
peaks_idx = argrelextrema(density(xs), np.greater)[0]
peaks_x = np.array([round (x) for x in xs[peaks_idx]])


plt.plot(xs, density(xs))
#plt.plot(peaks_x, density(peaks_x), 'ro')  # Plot peaks as red circles
for peak_x in peaks_x:
    plt.axvline(x=peak_x, color='red', linestyle='--')  # Vertical red line at peak
    plt.text(peak_x, density(peak_x), str(round(peak_x, 2)), color='red')


for x, peak_x in zip(density(peaks_x), peaks_x):
    plt.text(peak_x, x, str(round(peak_x, 2)), color='red')

plt.show()

print("X values for density peaks:")
print(peaks_x)


print("X values for density peaks:")
print(peaks_x)
exit()
# Plot the original data and the linear regression line
fig, ax = plt.subplots()
ax.scatter(prices_lm.index, prices_lm['close'], label='Data')
ax.plot(prices_lm.index, results.fittedvalues, 'r-', label='Linear Regression')

# Format x-axis ticks as dates
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Customize the date format if needed

plt.xlabel('Date')
plt.ylabel('close')
plt.legend()
plt.gcf().autofmt_xdate()  # Rotate and align the x-axis labels for better visibility
plt.show()

# Print the coefficients
print('Intercept:', intercept)
print('Slope:', slope)



exit ()
print(btc.columns)
print(btc.loc[btc.index[-300] : ,"d1_sma_8"].describe())
exit ()



""" 
btc = btc.dropna(axis=0)
btc['good_day'] = btc["close"] - btc['open']
btc['magnitude'] = btc['good_day'].copy()
btc['good_day'][btc['good_day'] <= 0] = 0
btc['good_day'][btc['good_day'] > 0] = 1
btc['magnitude'][btc['magnitude'] <= 0] = (btc['high'] - btc['low']) * 100 / btc['high']
btc['magnitude'][btc['magnitude'] > 0] = (btc['high'] - btc['low']) * 100 / btc['low']
btc['day_of_w'] = [weekDays[d.weekday()] for d in btc.index]
"""

def lucky_h(day, df = btc):
    day_index = df['day_of_w'] == day
    gd = btc.loc[day_index]["good_day"].sum()
    cp = btc.loc[day_index][["good_day" , "magnitude"]].copy()
    all= len(cp.index)
    cp["good_day"][cp['good_day'] == 0] = -1
    cp['tot'] =cp['good_day'] * cp['magnitude']
    return (gd, cp['tot'].sum(), all)

def volatile_day(day, df = btc):
    day_index = df['day_of_w'] == day
    vd = (btc.loc[day_index]["high"] - btc.loc[day_index]['low'])/((btc.loc[day_index]["open"]+btc.loc[day_index]["high"])/2)
    vdr = round(vd *100 , 2)
    return vdr.sum()


days = len(btc['good_day'])
good_days = btc['good_day'].sum()

#print(f'Avg P of up day since 2017: {good_days/days}\nTotal days: {days}')
top = btc['high'].idxmax()

dst = len(btc['good_day'].loc[top : ])
gdst = btc['good_day'].loc[top : ].sum()
gdfst = gdst/dst

print(f'Avg P of up day: {round(good_days/days, 2)}\nAvg P of good day since top : {round(gdfst,2)}\n')

bdst_filter_index = btc['good_day'].loc[top : ] == 0.0
gdst_filter_index = btc['good_day'].loc[top : ] == 1.0

up_mag_st = btc.loc[top : ][gdst_filter_index]["magnitude"].sum()
down_mag_st = btc.loc[top : ][bdst_filter_index]["magnitude"].sum()

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
for day in weekDays : 
    x, y, z = lucky_h(day=day)
    v = volatile_day(day=day)
   
    print(f'{day} >> {x} >>> {round(y,2)}>>>{z} ---- > lycky day coef: {x/z} ----> v is {round(v/z , 2)}')

# |~~~~~~~~~~~~~~~~~~ Study of the price variation relative to previous close (since pic)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
top = btc['high'].idxmax()
# seleting the data from the top to present
btc =  btc.loc[top : ]

privious_close = btc['close'].shift(1)
close_to_high_var = round(((btc['high'] - privious_close)/privious_close) * 100 , 2)
chanse_to_win_long = (sum(close_to_high_var > 1.39)/len(close_to_high_var)) 
close_to_low_var = round(((btc['low'] - privious_close)/privious_close) * 100 , 2)
chanse_to_win_short = (sum(close_to_low_var < -1.39)/len(close_to_low_var))
chance_to_swing = round(chanse_to_win_long * chanse_to_win_short , 2)

print(f'Chanse win : long -> {chanse_to_win_long} & short ->{chanse_to_win_short } : swing --> {chance_to_swing}')
sb.kdeplot(close_to_high_var , fill = True)
sb.kdeplot(close_to_low_var , fill = True)
plt.axvline(0 , linestyle = '--', color = 'red')
plt.show()
#|================================================================================================================|


plt.figure(figsize = (9,9))
sb.kdeplot(btc['var_prc'], fill = True)
sb.kdeplot(ada['var_prc']  , fill = True)
plt.axvline(0.95 , linestyle = '--', color = 'red')
plt.axvline(1.3 , linestyle = '--', color = 'blue')
plt.show()
'''
#mpf.plot(btc.loc[top : ], type='candle',mav=(12, 28, 78))
#btc.to_pickle("./data/btc_study.pkl")
#print('Cached at ./data/btc_study.pkl')
'''
plt.figure(figsize = (9,9))
sb.kdeplot(btc_df['var_prc'] , fill = True)
plt.show()
'''