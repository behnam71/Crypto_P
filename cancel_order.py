# -*- coding: utf-8 -*-
import os
import sys
from pprint import pprint

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt  # noqa: E402


exchange = ccxt.binance({
    'apiKey': 'SmweB9bNM2qpYkgl4zaQSFPpSzYpyoJ6B3BE9rCm0XYcAdIE0b7n6bm11e8jMwnI',  
    'secret': '8x6LtJztmIeGPZyiJOC7lVfg2ixCUYkhVV7CKVWq2LVlPh8mo3Ab7SMkaC8qTZLt',
    'enableRateLimit': True,
})
exchange.urls['api'] = exchange.urls['test']  # use the testnet
exchange.cancel_order('2023503', symbol='BTC/USDT')
