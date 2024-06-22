#from ws_harvy import TradeBot
from ws_sdt48_bot import Sdt48
from binance.enums import *
from time import sleep

bot = Sdt48(pair="MINAUSDT")
#harvy = TradeBot()
print(bot.get_price('MINAUSDT'))
#harvy.live_trade_pair()
bot.live_trade_pair()
