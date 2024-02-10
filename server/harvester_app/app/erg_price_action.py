import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import btalib as bl
import seaborn as sb
from sys import exit
from binance import Client
from datetime import datetime
import math

path = "./data/h/ERGUSTD_1h.csv"
def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='s')
    #df.set_index('date', inplace=True)
    #df.index = pd.to_datetime(df.index, unit='ms')
    return df

erg_df = import_coin_data(path)
print(erg_df.head())

