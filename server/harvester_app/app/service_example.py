from binance.websockets import ThreadedWebsocketManager
import signal
import time

class CryptoTrackerService:
    def __init__(self, symbol):
        self.symbol = symbol
        self.current_price = None
        self.twm = ThreadedWebsocketManager()
        self.is_running = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        print(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def handle_socket_message(self, msg):
        """Process incoming WebSocket messages"""
        try:
            if msg['e'] == '24hrTicker':
                self.current_price = float(msg['c'])
                print(f"{self.symbol} current price: {self.current_price}")
                # Here you can add your trading logic
            elif 'error' in msg:
                print(f"Error in WebSocket: {msg['error']}")
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def start(self):
        """Start the WebSocket connection"""
        self.twm.start()
        self.twm.start_symbol_ticker_socket(
            callback=self.handle_socket_message,
            symbol=self.symbol
        )
        self.is_running = True
        print(f"Started tracking {self.symbol}")
        
        # The WebSocket's event loop keeps the service running
        # No need for an explicit while loop
    
    def stop(self):
        """Stop the WebSocket connection"""
        if self.is_running:
            self.twm.stop()
            self.is_running = False
            print(f"Stopped tracking {self.symbol}")

def main():
    tracker = CryptoTrackerService("BTCUSDT")
    tracker.start()
    
    # The service will keep running until a signal is received
    # The ThreadedWebsocketManager handles its own event loop

if __name__ == "__main__":
    main()

"""
[Unit]
Description=Crypto Tracker Service
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
# This activates the conda environment before running the script
ExecStart=/bin/bash -c 'source /home/YOUR_USERNAME/miniconda3/etc/profile.d/conda.sh && conda activate fas && python /path/to/your/crypto_tracker.py'
WorkingDirectory=/path/to/your/script/directory
Restart=on-failure
RestartSec=10s
# Optional: Environment variables if needed
# Environment=VAR1=value1 VAR2=value2

[Install]
WantedBy=multi-user.target
"""