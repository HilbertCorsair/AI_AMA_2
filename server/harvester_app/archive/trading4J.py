import numpy as np
from time import sleep
import pandas as pd
from binance.client import Client
from binance import ThreadedWebsocketManager
import btalib as bl
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib.dates import date2num
#===================================================================================




coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]

mins = [
    15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
    869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230, 1.1
    ]
tops = [
    69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
    4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85, 19.23
    ]

mins_dict = dict(zip(coins, mins))
tops_dict = dict(zip(coins, tops))

of_interest =   ['BTC', 'ADA', 'MINA',    "PAXG", "AGIX", "DOT", "AR", "LINK"]
pairs = [f'{c}USDT' for c in of_interest]


BUSD_decs = [2, 4, 3, 0, 5, 2, 3, 2]
C_decs = [5, 1, 1, 4, 2, 3, 2, 3]

rounding_order_price = dict(zip(pairs, BUSD_decs))
rounding_order_crypro_amount = dict(zip(pairs, C_decs))

def get_price(pair):
    cli = unlock()
   # this gets the last traded price (no info on tranzaction type)
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, rounding_order_price[pair])
    return price


def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)
cli = unlock()

def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def plot_stoch_rsi_strategy (pair):

    buy = []
    sell = []
    switch = False
    fiat_bag = 100
    btc_bag = 0

    #df = pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{pair}.pkl").tail(last)
    data_file =  f'/home/honeybadger/projects/harvester/data/w/{pair}.csv'
    df = import_coin_data(data_file) #from old csvs !!! ALSO ADDS DATES AS INDEXES


    rsi = bl.rsi(df["close"], period=8)
    stoch_input = pd.DataFrame({
    'high': rsi.df['rsi'],
    'low': rsi.df['rsi'],
    'close': rsi.df['rsi']
    })
    
    stoch_rsi = bl.stoch(stoch_input, period = 3)

    for i in range(len(df.index)):
        if  not switch and stoch_rsi.df['k'][i] < 10 :# stoch_rsi.df['k'][i] > stoch_rsi.df['d'][i] and
            sell.append(float('nan'))
            buy.append(df['close'][i]) # <---- buying price
            btc_bag = fiat_bag / df["close"][i]
            fiat_bag = 0
            switch = True

        elif switch and stoch_rsi.df["k"][i] > 90 : # stoch_rsi.df['k'][i] < stoch_rsi.df['d'][i] and
            sell.append(df["close"][i]) # <----------- selling price
            buy.append(float('nan'))
            fiat_bag = btc_bag * df["close"][i]
            btc_bag = 0
            switch = False

        else : 
            buy.append(float('nan'))
            sell.append(float('nan'))
    
    final_evaluation = fiat_bag if fiat_bag !=0 else btc_bag * df["close"][-1]
    final_evaluation = round(final_evaluation -100 , 2)

    market  = round(((df["close"][-1] -  df["close"][0])/df["close"][0] ) * 100, 2)

    print(f"Congratulations your strategy made {final_evaluation}%")
    print(f"Meanwhile the market made {market}")


    # Top plot: Price with markers
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], vertical_spacing=0.1)

    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['close'], 
            mode='lines', 
            name='Price'
        ),
        row=1, col=1
    )

    # Buy markers
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=buy,
            mode="markers",
            marker=dict(color='green', symbol="triangle-up"),
            name='Buy'
        ),
        row=1, col=1
    )

    # Sell markers
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=sell,
            mode="markers",
            marker=dict(color='red', symbol="triangle-down"),
            name='Sell'
        ),
        row=1, col=1
    )

    # Bottom plot: Stochastic RSI
    fig.add_trace(
        go.Scatter(x=stoch_rsi.df.index, y=stoch_rsi.df['k'], mode='lines', name='K line'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=stoch_rsi.df.index, y=stoch_rsi.df['d'], mode='lines', name='D line'),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title='BTCUSDT Analysis',
        showlegend=True
    )

    # Update xaxis and yaxis properties if needed
    fig.update_xaxes(title_text='Date', row=2, col=1)
    fig.update_yaxes(title_text='Value', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=1, col=1)

    # Show plot
    fig.show()

def augment_MACD(df) :
    
    macd = bl.macd(df, pfast = 8, pslow = 63, psignal = 11)
    df = df.join(macd.df)
    x = np.array([date2num(d) for d in  df.index])
    y = df.loc[:,'histogram'].values
    df.loc[:, "momentum"] = np.gradient(x, y)
    return df 

def plot_macd_strat(pair, last = 1000):
    df = pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{pair}.pkl").tail(last)
    df = augment_MACD(df)


#==================================================================================
plot_stoch_rsi_strategy("BTCUSDT")