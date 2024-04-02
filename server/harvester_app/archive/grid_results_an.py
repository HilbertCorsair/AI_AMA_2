import pickle
import pandas as pd
import math


with open('./data/results_m3_rsi_macd_grid_search_H.pkl', 'rb') as f:
        data = pickle.load(f)

scores = [r for r in data[0]]
print(max(scores))
i = pd.Series(scores).idxmax()
best_combo = data[1][i]
rsi_buy = data.iloc[i][1][1][1]
rsi_sell = data.iloc[i][1][1][0]
print(best_combo)
print(rsi_buy, rsi_sell)

btc = pd.read_csv(f'./data/btc_2017H_bars.csv', names=['date', 'open', 'high', 'low', 'close'])
btc.set_index('date', inplace=True)
btc.index = pd.to_datetime(btc.index, unit='ms')
top = btc['high'].idxmax()
# seleting the data from the top to present
btc =  btc.loc[top : ]

start_price = btc["close"][0]
end_price = btc["close"][-1]

market_loss = round(((end_price-start_price)/start_price) * 100 , 2 )
model_gain_fiat = round((((end_price * 3.2) - start_price)/ start_price )*100,2)
print(f'MArket loss in fiat {market_loss}%\n')
print(f'In FIAT terms your model made {model_gain_fiat}%')