import AccountStatus
import oerations as ops
from binance.enums import *
from time import sleep
cli = AccountStatus.unlock()


pairs = [ "JUPUSDT", "PYTHUSDT"]
#print(round(AccountStatus.usdt_value, 2))

def get_price(pair):
   # this gets the last traded price (no info on tranzaction type)
    latest_price = cli.get_symbol_ticker(symbol = pair)
    price = float(latest_price['price'])
    price = round(price, 4)
    return price

targets= {"JUPUSDT": 0.5657,
          "PYTHUSDT": 0.5021
          }



while pairs:
    pj = get_price("JUPUSDT")
    pp = get_price("PYTHUSDT")

    if pj >= targets["JUPUSDT"]:
        ops.sell(pair="JUPUSDT", q = 2866, price=pj)
        pairs.remove('JUPUSDT')

    if pp >= targets["PYTHUSDT"]:
        ops.sell(pair="PYTHUSDT", q = 2866, price=pj)
        pairs.remove('PYTHUSDT')
    print(f'\nJUP ---> {pj}\nPYTH ---> {pp}\n')
    sleep(65)
    
    
