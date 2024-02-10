from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


coins = [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]
pairs = [f'{coin}BUSD' for coin in coins]

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

data_files = [f'./data/h/{pair}_1h2.csv' if not pair == 'ERGBUSD' else "./data/h/ERGUSTD_1h.csv" for pair in pairs]

def extract_avg_price(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    avg_price = (df["open"] + df["close"])/2
    return avg_price

c_data = [extract_avg_price(c) for c in data_files]
del c_data[17]
del coins[17]
c_data.pop()
coins.pop()

prices_df = pd.concat(c_data, axis=1 )
prices_df.columns =coins

for c in prices_df.columns:
    prices_df[c] = [(x - mins_dict[c]) / (tops_dict[c] - mins_dict[c]) for x in prices_df[c]]


scaler = StandardScaler()
prices_df_std = scaler.fit_transform(prices_df.transpose())

pca = PCA(n_components=7) # replace '2' with the number of principal components you want to extract
pca.fit(prices_df_std)

coins_pca = pca.transform(prices_df_std)
print(pca.explained_variance_ratio_)
print(pca.components_)

plt.scatter(coins_pca[:, 0], coins_pca[:, 1])
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()

fig = px.scatter_3d(x=coins_pca[:, 0], y=coins_pca[:, 1], z=coins_pca[:, 2], color = prices_df.columns)
fig.show()