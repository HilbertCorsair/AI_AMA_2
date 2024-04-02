import operations
import pandas as pd

class AccountStatus (operations.Ops):
    def __init__(self):
        super().__init__()
        
        self.stash = None
        self.hand = None
        self.usd_value = None
        self.prices_dict = None
        self.valuations = {}
        self.update_status()
        
    def update_status(self):
        cli = self.unlock() 
        ballance = cli.get_account()
        prices_dict = dict(zip(self.get_pairs(), [self.get_price(pair) for pair in self.get_pairs]))
        all_binance = ballance['balances']
        self.stash = pd.DataFrame([x for x in all_binance if float(x['free']) > 0 or float(x['locked']) > 0])

        for c in self.stash["asset"] :
            self.valuations[c] = self.get_price(f'{c}USDT') if c != "USDT" else 1

        self.valuations = pd.DataFrame({"asset": list(self.valuations.keys()), "price": list(self.valuations.values())})
        self.valuations = pd.merge(self.valuations, self.stash, on= "asset" , how="inner")

        def evaluate(row):
            return (float(row['free']) + float(row['locked']))* float(row["price"])
        
        self.valuations['value'] = [evaluate(r) for i, r in self.valuations.iterrows()]
        self.hand = self.valuations.iloc[self.valuations["value"].idxmax() , 0 ]
        self.usd_value = round(self.valuations.loc[:,'value'].sum(), 2)
      
 