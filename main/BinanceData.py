# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd

# -----------------------------------------------------------------------------
root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(
            symbol, timeframe, since, limit
        )
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')

def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    now = exchange.milliseconds()
    all_ohlcv = []
    fetch_since = since
    while fetch_since < now:
        ohlcv = retry_fetch_ohlcv(
            exchange, max_retries, symbol, timeframe, fetch_since, limit
        )
        fetch_since = (ohlcv[-1][0] + 1) if len(ohlcv) else (fetch_since + timedelta)
        all_ohlcv = all_ohlcv + ohlcv
        if len(all_ohlcv):
            print(len(all_ohlcv), 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
        else:
            print(len(all_ohlcv), 'candles in total from', exchange.iso8601(fetch_since))
    return exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)

def write_to_csv(filename, data):
    X = pd.DataFrame(data, columns = ['date', 'open', 'high', 'low', 'close', 'volume'])
    X['date'] = pd.to_datetime(X['date'], unit='ms')
    X.reset_index(drop=True)
    db = pd.read_csv('/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/database/db_{}'.format(filename), sep=',', low_memory=False, index_col=[0])
    db = pd.concat(
        [db, X],
        ignore_index=True, 
        sort=False
    )
    db.drop_duplicates(
        subset=['date'], 
        keep='first', 
        inplace=True
    )
    db = db.reset_index(drop=True)
    filename = "./database/db_{}".format(filename)
    db.to_csv(filename)

def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol, timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,  # required by the Manual
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(
        exchange, max_retries, symbol, timeframe, since, limit
    )
    # save them to csv file
    write_to_csv(filename, ohlcv)
    print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]))

def main():
    # Binance's BTC/USDT candles start on 2017-08-17
    print("BTC data fetching:")
    scrape_candles_to_csv('BTC.csv', 'binance', 3, 'BTC/USDT', '4h', '2016-08-17T04:00:00Z', 100)
    print("\nDOGE data fetching:")
    scrape_candles_to_csv('DOGE.csv', 'binance', 3, 'DOGE/USDT', '4h', '2018-08-17T04:00:00Z', 100)


if __name__ == "__main__":
    main()
