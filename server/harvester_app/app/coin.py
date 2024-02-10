from pycoingecko import CoinGeckoAPI
import datetime as dt
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
cg = CoinGeckoAPI()

market_data = cg.get_global()

# Extract total market cap
total_market_cap = market_data['total_market_cap']['usd']


symbols = [c.lower() for c in [
    "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
    "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
    ]]

coin_ids = {}
coins = cg.get_coins_markets(vs_currency="usd")
for coin in coins:
    if coin["symbol"] in symbols:
        coin_ids[coin["symbol"]] = coin["id"]

#cats = cg.get_coins_categories
print(coin_ids)


caps = {}
for lbl, id in coin_ids.items():
    caps[lbl] = cg.get_price(ids=id, vs_currencies='usd', include_market_cap = True)
print(caps)
