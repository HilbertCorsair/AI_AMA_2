import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException
import signal

class BinanceLiveTracker(object):
    """
    Trade bot that tracks the price and updates a trigger price
    """
    def __init__(self, symbols):
        self.symbols = symbols
        self.order_books = {symbol: {'bids': {}, 'asks': {}} for symbol in symbols}
        self.latest_prices = {symbol: None for symbol in symbols}
        self.client = None
        self.bm = None
        self.tasks = []
        self.running = True
        self.trigger = None

    async def initialize(self):
        self.client = await AsyncClient.create()
        self.bm = BinanceSocketManager(self.client)
        await self.get_snapshots()

    async def get_snapshots(self):
        for symbol in self.symbols:
            try:
                depth = await self.client.get_order_book(symbol=symbol, limit=1000)
                self.order_books[symbol]['bids'] = {float(price): float(qty) for price, qty in depth['bids']}
                self.order_books[symbol]['asks'] = {float(price): float(qty) for price, qty in depth['asks']}
                print(f"Order book snapshot fetched for {symbol}")
            except BinanceAPIException as e:
                print(f"Error fetching order book snapshot for {symbol}: {e}")


    async def start_websockets(self):
        self.tasks = [
            asyncio.create_task(self.depth_socket(symbol)) for symbol in self.symbols
        ] + [
            asyncio.create_task(self.ticker_socket())
        ]
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def depth_socket(self, symbol):
        async with self.bm.depth_socket(symbol) as stream:
            while self.running:
                try:
                    res = await asyncio.wait_for(stream.recv(), timeout=5)
                    self.process_depth_message(symbol, res)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error in depth socket for {symbol}: {e}")
                    if not self.running:
                        break
                    await asyncio.sleep(5)  # Wait before reconnecting

    async def ticker_socket(self):
        socket = self.bm.multiplex_socket([f"{symbol.lower()}@ticker" for symbol in self.symbols])
        async with socket as stream:
            while self.running:
                try:
                    res = await asyncio.wait_for(stream.recv(), timeout=5)
                    self.process_ticker_message(res)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Error in ticker socket: {e}")
                    if not self.running:
                        break
                    await asyncio.sleep(5)  # Wait before reconnecting

    def process_depth_message(self, symbol, msg):
        for bid in msg['b']:
            price, qty = float(bid[0]), float(bid[1])
            if qty == 0:
                self.order_books[symbol]['bids'].pop(price, None)
            else:
                self.order_books[symbol]['bids'][price] = qty

        for ask in msg['a']:
            price, qty = float(ask[0]), float(ask[1])
            if qty == 0:
                self.order_books[symbol]['asks'].pop(price, None)
            else:
                self.order_books[symbol]['asks'][price] = qty


    def process_ticker_message(self, msg):
        data = msg['data']
        symbol = data['s']
        self.latest_prices[symbol] = float(data['c'])

    def calculate_order_book_totals(self, symbol):
        bids = self.order_books[symbol]['bids']
        asks = self.order_books[symbol]['asks']

        total_bid_volume = sum(bids.values())
        total_ask_volume = sum(asks.values())
        total_bid_value = sum(price * quantity for price, quantity in bids.items())
        total_ask_value = sum(price * quantity for price, quantity in asks.items())

        return {
            'total_bid_volume': total_bid_volume,
            'total_ask_volume': total_ask_volume,
            'total_bid_value': total_bid_value,
            'total_ask_value': total_ask_value
        }

    def print_market_summary(self):
        print("\n" + "="*50)
        print("Binance Live Market Summary")
        print("="*50)

        for symbol in self.symbols:
            price = self.latest_prices[symbol]
            totals = self.calculate_order_book_totals(symbol)

            print(f"\n{symbol}:")
            print(f"  Current Price: {price:.8f}")
            print(f"  Total Bid Volume: {totals['total_bid_volume']:.4f}")
            print(f"  Total Ask Volume: {totals['total_ask_volume']:.4f}")
            print(f"  Total Bid Value: {totals['total_bid_value']:.4f}")
            print(f"  Total Ask Value: {totals['total_ask_value']:.4f}")

        print("\n" + "="*50)

    async def shutdown(self):
        print("Shutting down...")
        self.running = False
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.client.close_connection()
        print("Shutdown complete.")

async def print_summary_periodically(tracker):
    while tracker.running:
        await asyncio.sleep(10)
        tracker.print_market_summary()

async def main():
    symbols = ["BTCUSDT", "ADABTC", "MINABTC", "PAXGUSDT"]
    tracker = BinanceLiveTracker(symbols)
    
    await tracker.initialize()
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(tracker.shutdown()))
    
    # Start a background task to print market summary every 10 seconds
    summary_task = asyncio.create_task(print_summary_periodically(tracker))
    
    try:
        # Start WebSocket connections
        await tracker.start_websockets()
    except asyncio.CancelledError:
        pass
    finally:
        summary_task.cancel()
        await tracker.shutdown()

if __name__ == "__main__":
    asyncio.run(main())


