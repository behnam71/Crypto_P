# -*- coding: utf-8 -*-
import os
import sys
from pprint import pprint

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt


# instantiate exchanges
binance = ccxt.binance({
    'apiKey': 'SmweB9bNM2qpYkgl4zaQSFPpSzYpyoJ6B3BE9rCm0XYcAdIE0b7n6bm11e8jMwnI',  
    'secret': '8x6LtJztmIeGPZyiJOC7lVfg2ixCUYkhVV7CKVWq2LVlPh8mo3Ab7SMkaC8qTZLt',
})
binance.urls['api'] = binance.urls['test']  # use the testnet

try:
    # fetch account balance from the exchange
    binanceBalance = binance.fetch_balance()
    # output the result
    pprint(binanceBalance)

except ccxt.DDoSProtection as e:
    print(type(e).__name__, e.args, 'DDoS Protection (ignoring)')
except ccxt.RequestTimeout as e:
    print(type(e).__name__, e.args, 'Request Timeout (ignoring)')
except ccxt.ExchangeNotAvailable as e:
    print(type(e).__name__, e.args, 'Exchange Not Available due to downtime or maintenance (ignoring)')
except ccxt.AuthenticationError as e:
    print(type(e).__name__, e.args, 'Authentication Error (missing API keys, ignoring)')
