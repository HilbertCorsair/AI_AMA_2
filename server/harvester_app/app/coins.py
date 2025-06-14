
class Coins(object):
    """A simple class to store coin data """
    
    def __init__(self):

        self._coins = [
            "SOL", "PIXEL", "JUP", "WIF", "PYTH", "BTC", "ADA", "MINA", "PAXG", "AGIX", "DOT", "ALGO", "BNB", "MATIC", "LINK", "AR", "AAVE", "EGLD",
            "ETH", "SOL", "FIL", "AVAX", "APT", "REN", "RUNE", "GALA", "NEAR", "ATOM", "AMP", "ZIL", "NEO", "GRT", "RNDR", "UNI", "XMR", "SAND", "KAVA", "LOOM", "CAKE", "KSM", "ENJ", "ERG"
            ]

        self._mins = [
            10, 0.485, 0.02, 1, 0.38,  15460, 0.239, 0.24, 1606, 0.036,4,0.11,183, 0.315,5, 6,45.6, 32.31,
            869, 8, 2.4, 9.32, 1, 0.05, 1 , 0.015, 1.23, 5.5, 0.003, 0.0153, 5.94, 0.05, 0.275, 3.1, 96.5, 0.37, 5.1, 0.036, 2.48, 21.6, 0.230, 1.1
            ]
        self._tops = [
            255, 2, 2, 5, 2,   69020,3.1,6.68,2070, 0.95,55.13,2.95,693,2.94,53.1,91.24, 666.7,544.7,
            4867, 260, 238, 147, 60, 1.37, 21.3, 0.842, 20.6, 44.8, 0.09, 0.26, 141, 2.9, 8, 45, 520, 8.5, 9.22, 0.167, 44.2,  625, 4.85, 19.23
            ]
        self._of_interest =  ["BTC", "ADA"] # ['SOL','PIXEL', "JUP", "WIF", "PYTH"  ,'BTC', 'ADA', 'MINA', "PAXG", "AGIX", "DOT", "AR", "LINK"]

        self._BUSD_decs = [0, 5] # [ 2, 4, 4, 4, 4,    2, 4, 3, 0, 5, 2, 3, 3]
        self._C_decs =    [4, 0]#[ 2, 1, 1, 1, 1,    5, 1, 1, 4, 0, 3, 2, 2]

    def get_mins_tops(self):
        mins_dict = dict(zip(self._coins, self.mins))
        tops_dict = dict(zip(self._coins, self._tops))
        return mins_dict, tops_dict
    
    def get_pairs(self):
        pairs = [f'{c}USDT' for c in self._of_interest]
        return pairs
    
    def get_coins(self):
        return self._of_interest
    
    def get_roundings(self):
        rounding_order_price = dict(zip(self.get_pairs(), self._BUSD_decs))
        rounding_order_crypro_amount = dict(zip(self.get_pairs(), self._C_decs))
        return rounding_order_price, rounding_order_crypro_amount
    
        
