import pandas as pd
import cbpro as cb
#placing order in sandbox

key = '978638962982ec0fda25a57fdd55e4c6'
secret = 'Y+afeGgiZgHluJ4J0dR5NAjWNfuqsUdCdSjzTcaXmBo29rVIF3AiV3aDNHGN6RrS2FyJvLNJZYN98V9q1lVp9Q=='
passphrase = 'yb1sky8j2q'
api_url = 'https://api-public.sandbox.exchange.coinbase.com'

authanticated = cb.AuthenticatedClient(key, secret, passphrase, api_url = 'https://api-public.sandbox.exchange.coinbase.com')
print(authanticated.get_accounts())
order=authanticated.buy(product_id='BTC-EUR', order_type = 'limit',
                              price=22100, 
                              size='1.54',
                              )
print(order['id'])
authanticated.cancel_order(order["id"])
