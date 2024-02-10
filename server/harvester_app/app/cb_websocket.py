from socket import socket
import websocket, json

#open web socket to coinbase

def open_connection(ws):
    print('CONECTED !')
    subscribe_message = {
        'type': 'subscribe',
        'channels': [
            {'name': 'ticker',
            'product_ids': ['ADA-USD', 'BTC-USD']
             }
        ]
    }
    ws.send(json.dumps(subscribe_message))

def on_message(ws, message):
    data = json.loads(message)
    print(data)

def on_error(ws, err):
    print(err)

socket = "wss://ws-feed.exchange.coinbase.com"
ws = websocket.WebSocketApp(socket, on_open = open_connection, on_message =  on_message, on_error =  on_error)
ws.run_forever()

