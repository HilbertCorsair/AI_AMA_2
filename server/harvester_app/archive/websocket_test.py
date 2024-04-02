from binance import ThreadedWebsocketManager
import time

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return a , b

api_key, api_secret = unlock()

stash_usd = 1000
stash_ada = 0
def main():
    
    symbol = 'BTCBUSD'  # You can change this to monitor a different pair
    print_count = 0

    twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
    twm.start()

    def handle_socket_message(msg):
        nonlocal print_count
        if msg['e'] == 'trade':  # Trade event
           
            print(f"Time: {msg['T']}, Symbol: {msg['s']}, Last Buy Price: {msg['p']}, Last Sell Price: {msg['p']}")
            print_count += 1
            if print_count >= 15:
                twm.stop()

    twm.start_trade_socket(callback=handle_socket_message, symbol=symbol)
    twm.join()

if __name__ == "__main__":
    main()