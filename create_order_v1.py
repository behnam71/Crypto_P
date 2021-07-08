# -*- coding: utf-8 -*-
import asyncio
import os
import sys
from pprint import pprint

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt.async_support as ccxt  # noqa: E402


async def test():
    exchange = ccxt.binance({
        'apiKey': 'SmweB9bNM2qpYkgl4zaQSFPpSzYpyoJ6B3BE9rCm0XYcAdIE0b7n6bm11e8jMwnI',  
        'secret': '8x6LtJztmIeGPZyiJOC7lVfg2ixCUYkhVV7CKVWq2LVlPh8mo3Ab7SMkaC8qTZLt',
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        },
    })
    exchange.urls['api'] = exchange.urls['test']  # use the testnet

    response = None
    try:
        await exchange.load_markets()  # force-preload markets first
        #exchange.verbose = True  # this is for debugging
        symbol = 'BTC/USDT'       # change for your symbol
        amount = 0.001            # change the amount
        price = 34537.00          # change the price
        try:
            response = await exchange.create_limit_buy_order(symbol, amount, price)
        except Exception as e:
            print('Failed to create order with', exchange.id, type(e).__name__, str(e))
    except Exception as e:
        print('Failed to load markets from', exchange.id, type(e).__name__, str(e))
    #await exchange.close()
    return response

if __name__ == '__main__':
    pprint(asyncio.get_event_loop().run_until_complete(test()))
