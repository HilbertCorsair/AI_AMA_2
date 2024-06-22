
def strategy_output (df):
    buy = []
    sell = []

    c_dict = update_h_candles(pairs)
    prices_dict = dict(zip(pairs, [get_price(pair) for pair in pairs]))
    stash = {}# dict(zip(of_interest + ["USDT"], [float(d["free"]) for d in balance if d["asset"] in of_interest + ["USDT"]]))
    for c in of_interest:
        stash[c] = 0
    stash["USDT"] = 100
            
    # Determining the holding position
    valuations = {}
    for k, val in prices_dict.items():
        coin = [c for c in of_interest if c in k][0]
        valuations[coin] = val * stash[coin]
    valuations["USDT"] = stash["USDT"]
    valuations = pd.Series(valuations)
    
    bought = valuations.idxmax()
    usdt = bought == ("USDT")

    bags = {"fiat": 100}
    for c in of_interest:
        bags[c]= 0
    market_dict = {}

    for p in pairs:
        hist = c_dict[p][f"{p}_MACDh_12_26_9"].iloc[-1]
        momentum =c_dict[p][f"{p}_momentum"].iloc[-1]
        pi =c_dict[p][f"{p}_avg"].iloc[0]
        pf = c_dict[p][f"{p}_avg"].iloc[-1]

        mm = round((pf - pi)/pi, 2 )     
        
        print(f"{p}: {round(hist, 5)}: {round(momentum, 4)}, {round(pi,2)} --> {round(pf, 2)} | {mm}")
        market_dict[p] = (c_dict[p][f'{p}_avg'].iloc[-1] - c_dict[p][f'{p}_avg'].iloc[0] ) /c_dict[p][f'{p}_avg'].iloc[0]
    
    avg_mkt = sum(market_dict.values())/len(pairs)
    print(f"Average market move: {round(avg_mkt*100, 2)}" )

    latest_data = [df.iloc[-1, : ] for df in c_dict.values()]

    big_df = pd.concat(latest_data, axis= 0)
    
    histos = big_df.filter(like = "MACDh_12_26_9" )
    histos.sort_values(ascending=False, inplace= True)
    momentae = big_df.filter(like = "momentum" )
    momentae.sort_values(ascending = False, inplace = True)

    # take snapshot of the amplitude and direction of the price movement 
    
    pair = momentae.idxmax().split("_")[0]
    #pair_hist = histos[0]
    vector  = momentae.loc[f"{pair}_momentum"]

    #df = pd.read_pickle(f"/home/honeybadger/projects/harvester/data/h/pkls/{pair}.pkl").tail(last)
    data_file =  f'/home/honeybadger/projects/harvester/data/w/{pair}.csv'
    df = import_coin_data(data_file) #from old csvs !!! ALSO ADDS DATES AS INDEXES

    if  vector > 0.01 and usdt:
        # BUY !
        sell.append(float('nan'))
        buy.append(df['close'][i]) # <---- buying price
        btc_bag = fiat_bag / df["close"][i]
        fiat_bag = 0
        

    elif vector < 0 and not usdt:
        #SELL
        sell.append(df["close"][i]) # <----------- selling price
        buy.append(float('nan'))
        fiat_bag = btc_bag * df["close"][i]
        btc_bag = 0
        
    
    elif not usdt and  momentae.loc[f'{bought}USDT_momentum'] < 0 :
        # FLIPP
        sell.append(df["close"][i]) # <----------- selling price
        buy.append(float('nan'))
        fiat_bag = btc_bag * df["close"][i]
        btc_bag = 0


        pair = momentae.idxmax().split("_")[0]
        
        sell.append(float('nan'))
        buy.append(df['close'][i]) # <---- buying price
        btc_bag = fiat_bag / df["close"][i]
        fiat_bag = 0
    
    elif bought == "PAXG" and bought not in pair :
        sell_pair, q, p = pqp(bought)
        sell(sell_pair, q, p)

        stash["USDT"] = float(cli.get_asset_balance("USDT")["free"])
        
        q = round ((stash["USDT"]-5)/prices_dict[pair], rounding_order_crypro_amount[pair])
        p = round(prices_dict[pair], rounding_order_price[pair])
        print(f'buying {q} of {pair} for {p}')
        buy(pair, q, p)
        print(f"Flipped {bought} for {pair}")

    else:
        buy.append(float('nan'))
        sell.append(float('nan'))

    final_evaluation = fiat_bag if fiat_bag !=0 else btc_bag * df["close"][-1]
    final_evaluation = round(final_evaluation -100 , 2)

    market  = round(((df["close"][-1] -  df["close"][0])/df["close"][0] ) * 100, 2)

    print(f"Congratulations your strategy made {final_evaluation}%")
    print(f"Meanwhile the market made {market}")