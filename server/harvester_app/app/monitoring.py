from binance.client import Client
from time import sleep

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

pair = input("Pair ?\n")


while True:
    cli = unlock()
    latest_price = cli.get_symbol_ticker(symbol = pair)
    p = round(float(latest_price["price"]), 4)
    print(p)
    sleep(3*60)
    

