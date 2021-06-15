# -*- coding: utf-8 -*-
import asyncio
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt.async_support as ccxt  # noqa: E402


async def test():
    # instantiate exchanges
    exchange = ccxt.phemex({
        'enableRateLimit': True,
        'apiKey': '1f5e94b8-fc56-4e7b-aeeb-b66a46c0bd18',  
        'secret': 'dMtx9ZDyGXqXRhUwsz1INL8tVU8UZ5Xk9ZkZZ4QchfgyZThkNDhhOC1hNjE1LTQzOWMtYTE4MC1jYjVmMGMxZjQ2ZTA',
        'options': {
            'defaultType': 'swap',
        },

    })
    print(await exchange.fetch_balance({'currency':'BTC'}))
    await exchange.close()  # don't forget to close it when you're done
    return True

if __name__ == '__main__':
    print('CCXT version:', ccxt.__version__)
    print(asyncio.get_event_loop().run_until_complete(test()))
    
