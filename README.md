The AI-AMA app is not important right now It is the remake of an AI chat bot app but for now the reserves as a cash for various data analysis and  crypto trading bots. 
All the crypto related files ar in the server/harvester_app/app. The "garbage pile" file structure and lack of comets or documentation is by design. But that might change depending on how things go. There are many things I tried over the years but for now the focus is on the `server/harvester_app/app/shadow_stalker.py` . 

The script uses a Binance web-socket to monitor the price for a given pair and can place spot trading orders. The name comes from it's main functionality: monitoring the price, triggering and adjusting stop-loss orders while keeping the price limits private. 

Stop-loss orders are an important risk management tool but in crypto trading they are a vulnerability. They are public and susceptible to price manipulation especially on days with low volumes. By having a script that monitors the price "stop-loss", limits can remain private and also mobile. Unlike a classical stop-loss , the limit buy or sell orders are not immediately placed when the "stop-loss" is triggered. Instead the buy or sell price limits are updated only if the price moves in a favorable direction otherwise if the price reaches the limit a buy or sell spot order is placed.

The `PriceTracker` class is based on the `Ops` class that offers functionality related to Account information like valuation and current holdings (USDC or crypto). It is also based on the `RollingBufferMixin` class that acts like a "memory" buffer that stores a given number of recent price values (last n price values where for now n = 300) This `RollingBufferMixin` functionality is not yet in use. It will come into play later by providing important information about price volatility and  trend reversal signatures. 
The class also stores translations in the `transactions` dictionary. The transactions data is then stored in a .pkl file on disk at exit for analytics.

**Parameters :**
**pair** - The crypto trading pair. At the moment only vs USDC is supported but trading functionality vs BTC is being developed in the `activate_alt_btc_tracking`

**di, df, pi, pf, ...** - d is for date and p is for price. These parameters are the initial and final dates and prices that are taken from support and resistance lines drawn on [[https://www.tradingview.com/]]  these parameters are used to calculate the linear regression parameters (bata0 and bata1 for each line - support and resistance). The function `calculate_prices`takes the current timestamp and uses these parameters to determine the support and resistance values. It is these price values that are then used as a de facto stop-loss trigger with moving limit values. 

**prc_distance** - is the distance in percentage from the local min or max price (`record_price`) limit price that determines the placement of the buy or sell spot orders after the activation of the trigger. 

**momentum and update_frq** - they are no longer in use for now but momentum indicates the current trade direction and update_frq used be the number of seconds between api calls. In the absence of a stable network connection the script can run function via API calls by using the `api_cals` function. 

**How it's used** First the python modules have to be imported. Also the files coins, operations and buffer need to be present in the same directory as well as a nancy.txt file that only contains 2 lines of text coresponding to the binance API credentials. I know: only idiots put their api keys on github :)) it was an emergency and I changed the api key anyway.
For now the file is run as main from the terminal with the command: 
`python3 shadow_stalkter.py`
But first edit the file and enter the right parameters in the definition of the `main()` function that instantiates the PriceTracker class. 

The `start_trading`method opens the Web Socket to binance and subscribes to the ticker corresponding to the trading pair. It also triggers a call back function that deals with the message received from the WS. 

The `handle_ticker` takes the response from the web socket and acts like a switch between different trading strategies - methods that contain the actual reading logic.

There's no logging yet, just some printouts on the console. 

Have fun ! :) 

