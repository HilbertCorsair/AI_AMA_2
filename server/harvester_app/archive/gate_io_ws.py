# !/usr/bin/env python3
# coding: utf-8

import hashlib
import hmac
import json
import logging
import time

from websocket import WebSocketApp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GateWebSocketApp(WebSocketApp):

   def __init__(self, url, api_key, api_secret, **kwargs):
      super(GateWebSocketApp, self).__init__(url, **kwargs)
      self._api_key = api_key
      self._api_secret = api_secret

   def _send_ping(self, interval, event, payload):
      while not event.wait(interval):
         self.last_ping_tm = time.time()
         if self.sock:
            try:
               self.sock.ping(payload)
            except Exception as ex:
               logger.warning("send_ping routine terminated: {}".format(ex))
               break
            try:
               self._request("options.ping", auth_required=False)
            except Exception as e:
               raise e

   def _request(self, channel, event=None, payload=None, auth_required=True):
      current_time = int(time.time())
      data = {
         "time": current_time,
         "channel": channel,
         "event": event,
         "payload": payload,
      }
      if auth_required:
         message = 'channel=%s&event=%s&time=%d' % (channel, event, current_time)
         data['auth'] = {
            "method": "api_key",
            "KEY": self._api_key,
            "SIGN": self.get_sign(message),
         }
      data = json.dumps(data)
      logger.info('request: %s', data)
      self.send(data)

   def get_sign(self, message):
      h = hmac.new(self._api_secret.encode("utf8"), message.encode("utf8"), hashlib.sha512)
      return h.hexdigest()

   def subscribe(self, channel, payload=None, auth_required=True):
      self._request(channel, "subscribe", payload, auth_required)

   def unsubscribe(self, channel, payload=None, auth_required=True):
        self._request(channel, "unsubscribe", payload, auth_required)


def on_message(ws, message):
    # type: (GateWebSocketApp, str) -> None
    # handle message received
    logger.info("message received from server: {}".format(message))


def on_open(ws):
    # type: (GateWebSocketApp) -> None
    # subscribe to channels interested
    logger.info('websocket connected')
    ws.subscribe("options.contract_tickers", ['BTC_USDT-20211231-59800-C'], False)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)
    app = GateWebSocketApp("wss://op-ws.gateio.live/v4/ws",
                           "YOUR_API_KEY",
                           "YOUR_API_SECRET",
                           on_open=on_open,
                           on_message=on_message)
    app.run_forever(ping_interval=5)