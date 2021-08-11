# -*- coding: utf-8 -*-
import os
import sys
import time
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt  # noqa: E402


def main(symbol):
    print('Instantiating binance')
    # instantiate the exchange by id
    exchange = getattr(ccxt, 'binance')({
        'enableRateLimit': True, # https://github.com/ccxt/ccxt/wiki/Manual#rate-limit
    })

    if exchange.has['fetchTickers'] != True:
        raise ccxt.NotSupported ('Exchange ' + exchange.id + ' does not have the endpoint to fetch all tickers from the API.')

    # load all markets from the exchange
    markets = exchange.load_markets()
    try:
        tickers = exchange.fetch_tickers(symbol)
        for symbol, ticker in tickers.items():
            print(symbol, ticker['datetime'], ticker['ask'])
            return float(ticker['ask'])

    except ccxt.DDoSProtection as e:
        print(type(e).__name__, e.args, 'DDoS Protection (ignoring)')
    except ccxt.RequestTimeout as e:
        print(type(e).__name__, e.args, 'Request Timeout (ignoring)')
    except ccxt.ExchangeNotAvailable as e:
        print(type(e).__name__, e.args, 'Exchange Not Available due to downtime or maintenance (ignoring)')
    except ccxt.AuthenticationError as e:
        print(type(e).__name__, e.args, 'Authentication Error (missing API keys, ignoring)')


if __name__ == "__main__":
    main(symbol)
