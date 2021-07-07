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
    # 'options': {
    #     'defaultType': 'spot', // spot, future, margin
    # },
})

exchange.urls['api'] = exchange.urls['test']  # use the testnet
exchange.load_markets ()
#exchange.verbose = True # uncomment for debugging

symbol = 'BTC/USDT'
start_time = exchange.parse8601 ('2021-07-05T00:00:00')
now = exchange.milliseconds ()
day = 24 * 60 * 60 * 1000

all_trades = []
while start_time < now:
    print('------------------------------------------------------------------')
    print('Fetching trades from', exchange.iso8601(start_time))
    end_time = start_time + day
    trades = exchange.fetch_my_trades (symbol, start_time, None, {
        'endTime': end_time,
    })
    if len(trades):
        last_trade = trades[len(trades) - 1]
        start_time = last_trade['timestamp'] + 1
        all_trades = all_trades + trades
    else:
        start_time = end_time

print('Fetched', len(all_trades), 'trades')
for i in range(0, len(all_trades)):
    trade = all_trades[i]
    pprint(trade)
    print(i, trade['id'], trade['datetime'], trade['amount'])
