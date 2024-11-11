from ws_harvy import TradeBot
#from ws_sdt48_bot import Sdt48
from binance.enums import *
from time import sleep
from shadow_stalker import ShadowStalker

#bot = Sdt48(pair="MINAUSDT")
#harvy = TradeBot()
#print(bot.get_price('MINAUSDT'))
#harvy.live_trade_pair()
#bot.live_trade_pair()

#bot = ShadowStalker(pair='ADAUSDT', target_price=0.3233, stalking_distance=8)
#bot.stalk()

harvy = TradeBot("MINAUSDT")