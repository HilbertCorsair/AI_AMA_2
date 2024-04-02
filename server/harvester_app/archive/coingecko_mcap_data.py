from pycoingecko import CoinGeckoAPI
import datetime as dt
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time

cg = CoinGeckoAPI()
of_interest = ['bitcoin', "cardano", 'mina-protocol', "ergo"]

"""Function that takes a coin id and a number of days or max and fetches mcap historic date from coingecko 
then it writes the data to a csv file 
""" 
def fetch_market_cap_history(coin_id, days='max'):
    cg = CoinGeckoAPI()

    # Fetch historical market data for Bitcoin
    data = cg.get_coin_market_chart_by_id(id=coin_id, vs_currency='usd', days=days)

    # The market cap data is under the 'market_caps' key
    market_caps = data['market_caps']

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(market_caps, columns=['timestamp', 'market_cap'])
    return df

def get_mcap_data(coin_id):    
    df = pd.read_csv(f'data/d/{coin_id}_mcap_gecko.csv')
    # Convert timestamp to datetime
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.drop('timestamp', axis=1, inplace=True)
    return df



def update_mcap_history (coin_id):
    current_time_milliseconds = int(time.time() * 1000)
    current_data = pd.read_csv(f'data/d/{coin_id}_mcap_gecko.csv')
    #current_data = current_data.sort_values(by = 'timestamp', ascending = True) 
    latest_timestamp = current_data['timestamp'].iloc[-1]
    days = round((current_time_milliseconds - latest_timestamp) /(1000*60*60*24))
    print(f'{days} days since previous {coin_id} update')
    new_data = fetch_market_cap_history(coin_id, days = days)
    current_data = pd.concat([current_data, new_data], ignore_index=True)
    current_data.to_csv(f'data/d/{coin_id}_mcap_gecko.csv', index=False)
    print(f'{coin_id} updated!')  


#update_mcap_history
#for coin in of_interest:
#    update_mcap_history(coin)

# Fetch and print the DataFrame
ada_market_cap = get_mcap_data("cardano")

btc_market_cap = get_mcap_data("bitcoin")
erg_mc = get_mcap_data("ergo")
mina_mc = get_mcap_data('mina-protocol')
tail  = mina_mc.shape[0]
print(mina_mc.tail(5))
print(btc_market_cap.head(5))
print(ada_market_cap.tail(5))
print(erg_mc.tail())
exit()


#ada_market_cap = ada_market_cap.sort_values(by = "date", ascending = True)
#btc_market_cap = btc_market_cap.sort_values(by = "date", ascending = True)
#erg_mc = erg_mc.sort_values(by = "date", ascending = True)
#mina_mc = mina_mc.sort_values(by = "date", ascending = True)

fig = make_subplots(rows=3, cols=1, subplot_titles=("Bitcoin Price", "Bitcoin Market Cap", "Bitcoin Total Volume"))

# Add traces
fig.add_trace(go.Scatter(x = btc_market_cap['date'], y = np.log10(btc_market_cap['market_cap']), name ='Market Cap', mode = 'lines', line=dict(color='green')), row=2, col=1)
fig.add_trace(go.Scatter(x = ada_market_cap['date'], y = np.log10(ada_market_cap['market_cap']), name = 'Market Cap', mode = 'lines', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x = erg_mc['date'], y = np.log10(erg_mc['market_cap']), name = 'Market Cap', mode = 'lines', line=dict(color='black')), row=2, col=1)
fig.add_trace(go.Scatter(x = mina_mc['date'], y = np.log10(mina_mc['market_cap']), name = 'Market Cap', mode = 'lines', line=dict(color='red')), row=2, col=1)

# Update layout
fig.update_layout(height=800, title_text="Bitcoin Historical Data", showlegend=False)

# Show plot
fig.show()
