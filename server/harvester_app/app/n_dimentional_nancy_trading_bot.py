from binance import Client
import pandas as pd
import numpy as np
import btalib as bl
from time import sleep
from binance.exceptions import BinanceAPIException, BinanceOrderException

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<| GO! |>~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data_files = [
    "./data/h/ADAUSDT_1h.csv",
    "./data/h/BTCUSDT_1h.csv",
    "./data/h/MINAUSDT_1h.csv",
    "./data/h/PAXGUSDT_1h.csv",
    "./data/h/ADABTC_1h.csv",
    "./data/h/MINABTC_1h.csv",
    "./data/h/PAXGBTC_1h.csv"
    ]


of_interest = ['BTC', 'ADA', 'MINA', "PAXG"]
pairs = ["ADABUSD" ,"BTCBUSD", "MINABUSD", "PAXGBUSD", "ADABTC", "MINABTC", "PAXGBTC"]
BUSD_decs = [3, 1, 3, 1, 7, 7, 5]
rounding_fiat = dict(zip(pairs, BUSD_decs))
C_decs = [1, 4, 1, 2, 6, 6, 3]
soldrounding_order_crypro_amount = dict(zip(pairs, C_decs))

def unlock (fname = 'nancy.txt'):
    with open(fname) as f:
        lines = f.readlines()
    a = lines[0].splitlines()[0]
    b = lines[1]
    return Client(a , b)

def import_coin_data(pth):
    df = pd.read_csv(pth, names=['date', 'open', 'high', 'low', 'close'])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    #df.set_index('date', inplace=True)
    #df.index = pd.to_datetime(df.index, unit='ms')
    return df

def trim (coin_df):
    top = coin_df['high'].idxmax()
    trimmed_df = coin_df.loc[top: ] 
    return trimmed_df
def delta (arr):
    s = arr[-1] - arr[0]
    return s 

def norm_augment(df) :
    
    top_price = max(df.loc[:,'high'])
    botom_price = min(df.loc[:,"low"])
    
    rng = range(len(df.index))
    #first : store the price in the avg col 
    df.loc[:,'avg'] = [(df["open"][i] + df["close"][i])/2 for i in rng]
    # !!! SUPER IMPORTANT price values are overwritten with their normalized values (min - max normalisation)
    # this step makes it possible to compare the MACD values across all coins 
    # MACD values are not calculated on the absolute prices but on their respective normalized values
    df.loc[:,"open"] = [(df["open"][i] - botom_price)/(top_price - botom_price) for i in rng]
    
    df.loc[:,"high"] = [(df["high"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"low"] = [(df["low"][i] - botom_price)/(top_price - botom_price) for i in rng]
    df.loc[:,"close"] = [(df["close"][i] - botom_price)/(top_price - botom_price) for i in rng]
    
    macd = bl.macd(df, pfast = 12, pslow = 26, psignal = 10)
    df = df.join(macd.df)
    df.loc[:,'momentum'] = df.loc[:,'histogram'].rolling(2, closed = "left").apply(delta)
    cutoff = len(df.index) - 339
    return df.iloc[cutoff: , :]

# Updating the Coins data dictionary
newts=[]
def update_pair (pair):
    cli = unlock()
    bars = cli.get_historical_klines(pair, "1h", timestamp)
    #create empty dataframe
    df = pd.DataFrame( columns= ['date', 'open', 'high', 'low', 'close'])
    for bar in bars:
        # add each row to the dataframe
        df.loc[len(df.index)] = [float(x) for x in bar[0:5]]
    
    newts.append(int(df.iloc[-1,0]))
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    return df

def get_price(pair):
    cli = unlock()
    candles = cli.get_klines(symbol=pair, interval=Client.KLINE_INTERVAL_30MINUTE)
    lc= candles[-1]
    df = pd.DataFrame( columns= ['date', 'open', 'high', 'low', 'close'])
    df.loc[len(df.index)] = [float(x) for x in lc[0:5]]
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index, unit='ms')
    avg = float((df["open"] + df["close"]) / 2)
    r = rounding_fiat[pair] 
    avg = round(avg, r) if pair not in ['BTCBUSD', 'PAXGBUSD'] else int(avg)
    return avg

def sell( pair, q , price):
    cli = unlock()
    try: 
        sell_limit = cli.create_order(
            symbol = pair,
            side='SELL',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        print(f" Placing SELL {pair} order ...")
        while cli.get_open_orders(symbol = pair ):
            if c1 < 1 :
                print("waiting for order to be filled ...")
                c1 += 1
            
            c1 += 1
            if c1 == 50:
                cli.cancel_order(symbol = pair, orderID = sell_limit['orderId'])
                break

            else : 
                sleep(15)
        print("DONE!")

    except BinanceAPIException as e:
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
def buy( pair, q , price):
    cli = unlock()
    try: 
        buy_limit = cli.create_order(
            symbol = pair,
            side='BUY',
            type='LIMIT',
            timeInForce='GTC',
            quantity = q,
            price = price)

        c1 = 0
        # While loop waits 50 min for order to fill than cancels 
        print(f'Placeing BUY order...')
        while cli.get_open_orders(symbol = pair ):
            if c1 < 1 :
                print(f"waiting for BUY order {pair} to be filled ...")
                c1 += 1

            c1 += 1
            if c1 == 50:
                cli.cancel_order(symbol = pair, orderID = buy_limit['orderId'])
                break

            else : 
                sleep(15)
        print("DONE")

    except BinanceAPIException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)
        
    except BinanceOrderException as e:
        # error handling goes here
        print(f'Oder failed to pass for q {q}, with price {price}, for {pair}')
        print(e)

#Gunction that checks the account and returns the lable of the max holding 
def get_holding_and_prices(stash_dict):
    prices = [get_price(f'{coin}BUSD') for coin in of_interest]
    prices = dict(zip(of_interest, prices ) )
    stash_val = {}
    # Collecting BUSD values for coins in account (stash_dict)
    for k,v in prices.items():
        stash_val[k] = v * stash_dict[k]
    stash_val["BUSD"] = stash_dict['BUSD']

    max_key = max(stash_val, key=stash_val.get)
    return max_key, prices

def mac_4D_strat (df):
    cli = unlock()
    print("initiating ...")
    histos = df.filter(like = "histogram" )
    momentae = df.filter(like = "momentum" )
    lbls = ["ADA", "MINA", "BTC", "PAXG"]

    balance = [cli.get_asset_balance(c)for c in ['BTC', 'ADA', 'MINA', "PAXG", "BUSD"]]
    stash = dict(zip(of_interest + ["BUSD"], [float(d["free"]) for d in balance]))
    print("Checking evaluations ... ")
    max_val, prices = get_holding_and_prices(stash)
    bought = False if max_val == 'BUSD' else max_val
    
    #bought = False if stash['BUSD'] > 50 else 
    bought_the_dipp = False
    #round(stash["BUSD"])
    
    ind = df.name
    print(f'Starting trading session : {ind}')

        # take snapshot of the amplitude and direction of the price movement 

    snapshot = {
        "hist_min_idx" : histos.idxmin(),
        "hist_min_val" : histos.min(),
        "hist_max_idx" : histos.idxmax(),
        "hist_max_val" : histos.max(),
        "mom_min_idx" : momentae.idxmin(),
        "mom_min_val" : momentae.min(),
        "mom_max_idx" : momentae.idxmax(),
        "mom_max_val" : momentae.max()
    }

    if snapshot["hist_max_val"] > 0 and not bought :

        #chech momentum for the pair
        pair = snapshot["hist_max_idx"].split("_")[0]
        vector  = momentae.loc[f"{pair}_momentum"]
        if vector > 0:
            #Check coin to buy and get price
            coin = [c for c in lbls if c in pair]
            price = prices[coin[0]]
            
            #define the quantity to purchase 
            q = round((stash["BUSD"] * 0.95) /price , soldrounding_order_crypro_amount[pair]  ) if pair in ["BTCBUSD", "PAXGBUSD"] else int((stash["BUSD"] * 0.95))
            print(stash["BUSD"]> q*price)
            
            print(f"buing...{pair}")

            # BUY !
            """
            try: 
                buy_limit = cli.create_order(
                    symbol=pair,
                    side='BUY',
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity = q,
                    price = price)

                ord = cli.get_open_orders(symbol=pair)
                c1 = 0
                # While loop waits 50 min for order to fill than cancels 
                while ord:
                    print(f"Waiting for BUY {pair} order to execute ...")
                    c1 += 1
                    if c1 == 50:
                        cli.cancel_order(symbol = pair, orderId = buy_limit['orderId'])
                        break

                    else : 
                        sleep(15)

                
                

            except BinanceAPIException as e:
                # error handling goes here
                print(e)
                
            except BinanceOrderException as e:
                # error handling goes here
                print(e)
            """
            buy(pair, q, price)
            bought = [str for str in lbls if str in pair][0]
            print(f"1. Bought {q} on  {pair} pair : {ind}")
                


    if snapshot["hist_max_val"] < 0 and not bought :
            # FIND OUT IF any vectors are positive and buy the strongest one 
        pair = snapshot["hist_min_idx"].split("_")[0]
        vector =  momentae.loc["MINABUSD_momentum", "ADABUSD_momentum", "BTCBUSD_momentum", "PAXGBUSD_momentum"].max()
        strongest = momentae.loc["MINABUSD_momentum", "ADABUSD_momentum", "BTCBUSD_momentum", "PAXGBUSD_momentum"].idxmax()
        strongest = strongest.split("_")[0]
        print("Market checing ...")
        if vector > 0:

            coin = [c for c in lbls if c in strongest]
            price = prices[coin[0]]
            #define the quantity to purchase 
            q = round(int(stash["BUSD"] * 0.95)  / price , soldrounding_order_crypro_amount[strongest]  ) if pair in ["BTCBUSD", "PAXGBUSD"] else int(stash[bought]* 0.995)
            print(f"Buying {strongest}")

            buy(strongest, q, price)
            bought = [str for str in lbls if str in strongest][0]
            bought_the_dipp = False if bought == "PAXG" or pair != strongest else True
            print(f"2. Bought {q} on  {strongest} pair : {ind}")


            """
            try: 
                buy_limit = cli.create_order(
                    symbol = strongest,
                    side='BUY',
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity = q,
                    price = price)

                ord = cli.get_open_orders(symbol = strongest )
                c1 = 0
                # While loop waits 50 min for order to fill than cancels 
                while ord:
                    print(f"Waiting for BUY {strongest} order to execute ...")
                    c1 += 1
                    if c1 == 50:
                        cli.cancel_order(strongest, buy_limit ['orderId'])
                        break

                    else : 
                        sleep(15)
                

            except BinanceAPIException as e:
                # error handling goes here
                print(e)

            except BinanceOrderException as e:
                # error handling goes here
                print(e)
            """

    
    elif bought_the_dipp :
        # if bought is top pair and the momentum shifts sell to fiat (exit)
        pair = snapshot["hist_max_idx"].split("_")[0]
        vector  = momentae.loc[ f"{pair}_momentum"]
        if  bought in pair and vector < 0 :
            # TAKE PROFIT

            coin = [c for c in lbls if c in pair]
            price = prices[coin[0]]
            #define the quantity to purchase 
            q = round(stash[bought]* 0.995 , soldrounding_order_crypro_amount[pair]  ) if pair in ["BTCBUSD", "PAXGBUSD"] else int(stash[bought]* 0.995)

            sell(pair, q, price)
            bought = False
            bought_the_dipp = False
            print(f"3. sold {q} on  {pair} pair for {stash['BUSD']} bought ON DISCOUNT ! : {ind}")

            """

            try: 
                sell_limit = cli.create_order(
                    symbol = pair,
                    side='SELL',
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity = q,
                    price = price)

                ord = cli.get_open_orders(symbol = pair )
                c1 = 0
                # While loop waits 50 min for order to fill than cancels 
                while ord:
                    print(f"Waiting for BUY {pair} order to execute ...")
                    c1 += 1
                    if c1 == 50:
                        cli.cancel_order(symbol = pair, orderID = sell_limit['orderId'])
                        break

                    else : 
                        sleep(15)
                

            except BinanceAPIException as e:
                # error handling goes here
                print(e)

            except BinanceOrderException as e:
                # error handling goes here
                print(e)
            
            """

    #SWITCH AT CROSS ! 
    elif bought : #and not bought_the_dipp:
        # check top pair
        pair = snapshot["hist_max_idx"].split("_")[0]

        if not bought in pair:
            #check if BTC is on top 
            if pair == "BTCBUSD":
                print(bought)
                # SWITCH to BTC
                sell_pair = f'{bought}BUSD'

                #place COINBTC SELL ORDER OR COIN BUSD sell and BUSDBTC buy
                coin = [c for c in lbls if c in sell_pair]
                price = prices[coin[0]]
                q = round(stash[bought] *0.99 , soldrounding_order_crypro_amount[sell_pair]) if coin == "PAXG" else int(stash[bought] *0.995)
                sell(sell_pair, q, price)
                
                # BUY !
                #place buy order
                coin = [c for c in lbls if c in pair]
                price = prices[coin[0]]
                q = round((stash["BUSD"] * 0.99)  / price  , soldrounding_order_crypro_amount[pair]) 
                buy(pair, q, price)

                print(f'4. Flipped {bought} for BTC on {ind}!')
                bought = "BTC"
    

            elif bought == "BTC" and not pair == "BTCBUSD":
                # Swithc from BTC to new top coin
                top_coin = [str for str in ["ADA", "MINA", "PAXG"] if str in pair][0]
                
                # SELL BTC
                price = prices['BTC']
                q = round(stash["BTC"]* 0.995 , soldrounding_order_crypro_amount['BTCBUSD'])
                sell("BTCBUSD", q, price)
                
                # Buy COIN
                pair = f'{top_coin}BUSD'
                price = prices[top_coin]
                q = round((stash["BUSD"] * 0.99 )/price, soldrounding_order_crypro_amount[pair])if pair in ["BTCBUSD", "PAXGBUSD"] else int(stash[bought]*0.995)

                bought = top_coin
                print(f'5. Flipped BTC for {bought} on {ind}')
                
            else:
                # COIN to COIN FLIP
                top_coin = [str for str in ["ADA", "MINA", "PAXG"] if str in pair][0]
                buy_pair = f'{top_coin}BUSD'
                sell_pair = f'{bought}BUSD'

                # SELL COIN 1 
                coin = [c for c in lbls if c in sell_pair]
                price = prices[coin[0]]
                q1 = round(stash[bought]* 0.995 , soldrounding_order_crypro_amount[sell_pair]) if sell_pair in ["BTCBUSD", "PAXGBUSD"] else int(stash[bought]*0.995)  
                sell(sell_pair, q1, price)


                coin = [c for c in lbls if c in buy_pair]
                price = prices[coin[0]]
                q2 = round(( stash["BUSD"]* 0.995 ) / price, soldrounding_order_crypro_amount[buy_pair] ) if buy_pair in ["BTCBUSD", "PAXGBUSD"] else int(stash[bought]*0.995)
                buy(buy_pair, q2, price)

                print(f'6) Converted {q1} {bought} to{q2} {top_coin}')
                print(f'Stash value : {round(q2 * price, 2)} on {ind}')
                bought = top_coin

        # boght in pair but ...
        if  not  pair == "BTCBUSD" and bought == "BTC":
            # Swithc from BTC to new top coin

            top_coin = [str for str in ["ADA", "MINA", "PAXG"] if str in pair][0]
            
            # SELL BTC
            price = prices["BTC"]
            q = round(stash["BTC"]* 0.99 , soldrounding_order_crypro_amount['BTCBUSD'])
            sell("BTCBUSD", q, price)
            
            # Buy COIN
            pair = f'{top_coin}BUSD'
            price = prices[top_coin]
            q = round((stash["BUSD"]* 0.995 )/price, soldrounding_order_crypro_amount[pair]) if pair == ["PAXGBUSD"] else int(stash[bought]*0.995)   

            bought = top_coin
            print(f'7. Flipped BTC for {bought} on {ind}')
            
        
        else:
            #Your're on top check momentum
            vector  = momentae.loc[f"{pair}_momentum"]
            if vector < 0 :
                # SELL the TOP !

                coin = [c for c in lbls if c in pair]
                price = prices[coin[0]]
                q = round(stash[bought]*0.995, soldrounding_order_crypro_amount[pair]) if pair in ["BTCBUSD", "PAXGBUSD"] else int(stash[bought]*0.995)  
                sell(pair, q, price)
                print(f'8) Just sold the topp {pair} on {ind} !!!')
                bought = False 


#=======================================================================================================================
print("importing file data")
c_data = [(import_coin_data(c)) for c in data_files]

c_data = dict(zip(pairs, c_data))

timestamp = str(1674201600000)
print("OLD TS IS >", timestamp)
# Update candls 


h = 0
while h < 25:
    
    for p in pairs:
        new_df = update_pair(p)
        c_data[p] = pd.concat([c_data[p], new_df], axis=0)


    augmented_data = [norm_augment(c_data[p]) for p in pairs]

    # Change the column names of the dataframes to include the pair so thei can be joined
    for i in range(len(pairs)):
        cols = augmented_data[i].columns
        augmented_data[i].columns = [ f'{pairs[i]}_{col}' for col in cols ]

    augmented_data = pd.concat(augmented_data, axis= 1 )
    last_candle = augmented_data.iloc[-1 , :] 
    timestamp = str(newts[0])

    print("Appling strategy...")
    mac_4D_strat(last_candle)
    h += 1
    print("Comeback in 1h ")
    sleep (3420) 


# TO DO :
# ELIMINATE DUPLIVATE ENTRIES 
# Check if new candle = old one 
