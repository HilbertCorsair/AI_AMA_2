
from binance.websockets import ThreadedWebsocketManager
import time

class CryptoTracker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.current_price = None
        self.twm = ThreadedWebsocketManager()
        self.is_running = False
    
    def handle_socket_message(self, msg):
        """Process incoming WebSocket messages"""
        try:
            if msg['e'] == '24hrTicker':
                self.current_price = float(msg['c'])
                print(f"{self.symbol} current price: {self.current_price}")
            elif 'error' in msg:
                print(f"Error in WebSocket: {msg['error']}")
        except Exception as e:
            print(f"Error processing message: {e}")
            print(f"Message received: {msg}")
    
    def start(self):
        """Start the WebSocket connection"""
        self.twm.start()
        self.twm.start_symbol_ticker_socket(
            callback=self.handle_socket_message,
            symbol=self.symbol
        )
        self.is_running = True
        print(f"Started tracking {self.symbol}")
    
    def stop(self):
        """Stop the WebSocket connection"""
        self.twm.stop()
        self.is_running = False
        print(f"Stopped tracking {self.symbol}")

def main():
    # Create a tracker for BTCUSDT
    tracker = CryptoTracker("BTCUSDT")
    
    try:
        # Start the WebSocket
        tracker.start()
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Program manually stopped")
    finally:
        # Ensure WebSocket is closed properly
        if tracker.is_running:
            tracker.stop()

if __name__ == "__main__":
    main()

"""This example:
Creates a clean class structure for tracking a crypto pair
Properly initializes the ThreadedWebsocketManager
Implements proper error handling in the WebSocket callback
Provides clean start/stop methods
Ensures the WebSocket is properly closed when the program exits
This is a minimal implementation that you can expand with your specific trading logic as needed."""