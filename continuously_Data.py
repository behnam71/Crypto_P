# -*- coding: utf-8 -*-
import asyncio
import os
import sys
import pandas as pd
from pprint import pprint

from ta import add_all_ta_features

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root + '/python')

import ccxt.async_support as ccxt  # noqa: E402


# this example shows how to fetch OHLCVs continuously, see issue #7498
# https://github.com/ccxt/ccxt/issues/7498
async def fetch_ohlcvs_continuously(exchange, timeframe, symbol, windows_s, fetching_time):
    print(exchange.id, timeframe, symbol, 'starting')
    all_ohlcvs = []
    limit = 1
    duration_in_seconds = exchange.parse_timeframe(timeframe)
    duration_in_milliseconds = duration_in_seconds * 1000
    now = exchange.milliseconds()
    end_time = now + fetching_time
    while now < end_time:
        since = int(now / duration_in_milliseconds) * duration_in_milliseconds
        time_to_wait = duration_in_milliseconds - now % duration_in_milliseconds + 10000  # +10 seconds buffer
        print(exchange.id, timeframe, symbol, 'time now is', exchange.iso8601(now))
        print(exchange.id, timeframe, symbol, 'time to wait is', time_to_wait / 1000, 'seconds')
        print(exchange.id, timeframe, symbol, 'sleeping till', exchange.iso8601(now + time_to_wait))
        await exchange.sleep(time_to_wait)
        print(exchange.id, timeframe, symbol, 'done sleeping at', exchange.iso8601(exchange.milliseconds()))
        while True:
            try:
                ohlcvs = await exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                break
            except Exception as e:
                print(type(e).__name__, e.args, str(e))  # comment if not needed
                #break | pass - to break it | to do nothing and just retry again on next iteration
                # or add your own reaction according to the purpose of your app
        print(exchange.id, timeframe, symbol, 'fetched', len(ohlcvs), 'candle(s), time now is', exchange.iso8601(exchange.milliseconds()))
        print(exchange.id, timeframe, symbol, 'all candles:')
        all_ohlcvs += ohlcvs
        dataset = pd.DataFrame()
        if len(all_ohlcvs) == windows_s:
            df = pd.DataFrame(all_ohlcvs, columns =['date', 'open', 'high', 'low', 'close', 'volume'])
            df.reset_index(drop=True)
            dataset = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=False)
            dataset = dataset[['date', 'open', 'high', 'low', 'close', 'volume', 'volume_adi', 'volume_obv', 'volume_mfi', 
                               'volatility_atr', 'volatility_bbm', 'volatility_bbh','volatility_bbl', 'trend_macd', 
                               'trend_macd_signal', 'trend_macd_diff', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx', 
                               'trend_adx_pos', 'trend_adx_neg', 'trend_cci', 'trend_ichimoku_conv', 'trend_ichimoku_base', 
                               'trend_ichimoku_a', 'trend_ichimoku_b', 'trend_psar_up', 'trend_psar_down', 'momentum_rsi', 
                               'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr']] # chart data
            dataset.to_csv("data1.csv")

        for ohlcv in all_ohlcvs:
            print(" ", exchange.iso8601(ohlcv[0]), ohlcv[1:])
        now = exchange.milliseconds()
    return {symbol: all_ohlcvs}


async def fetch_all_ohlcvs_continuously(loop, exchange_id, timeframe, symbols, windows_s, fetching_time):
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True, 'asyncio_loop': loop})
    input_coroutines = [fetch_ohlcvs_continuously(exchange, timeframe, symbol, windows_s, fetching_time) for symbol in symbols]
    results = await asyncio.gather(*input_coroutines, return_exceptions=True)
    await exchange.close()
    return exchange.extend(*results)


def fetchData(timeframe, symbols, windows_s):
    print('CCXT version:', ccxt.__version__)

    exchange_id = 'binance'
    fetching_time = 15 * 60 * 1000 # stop after 15 minutes (approximately 4 iterations)
    loop = asyncio.get_event_loop()
    coroutine = fetch_all_ohlcvs_continuously(loop, exchange_id, timeframe, symbols, windows_s, fetching_time)
    results = loop.run_until_complete(coroutine)
    pprint(results)
