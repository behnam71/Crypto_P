import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo

import pandas as pd
import ta
from IPython.display import display

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio

import tensortrade.env.default as default
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger, ScreenLogger


def load_csv(filename):
    df = pd.read_csv(filename, low_memory=False, index_col=[0])
    """
    df = pd.read_csv(filename, skiprows=1)
    df.drop(columns=['unix', 'symbol', 'Volume BTC', 'tradecount'], inplace=True)
    df = df.rename({Volume USDT: "volume"}, axis=1)

    temp = ''
    # Fix timestamp form "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    for i in range(0, len(df)):
        if df.loc[i, 'date'][-2:]!='AM' and df.loc[i, 'date'][-2:]!='PM':
            if df.loc[i, 'date'] != temp:
                d = datetime.strptime(df.loc[i, 'date'], "%Y-%m-%d %H:%M:%S")
                df.loc[i, 'date'] = d.strftime("%Y-%m-%d %I-%M-%S %p")
                temp = df.loc[i, 'date']
        else:
            df.loc[i, 'date'] = df.loc[i, 'date'][:13] + '-00-00 ' + df.loc[i, 'date'][-2:]

	# Convert the date column type from string to datetime for proper sorting.
    df['date'] = pd.to_datetime(df['date'])
    
    # Make sure historical prices are sorted chronologically, oldest first.
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Format timestamps as you want them to appear on the chart buy/sell marks.
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')
    df.to_csv(filename)
    """
    return df


def main():

    register_env("TradingEnv", create_env)

    analysis = tune.run(
        "PPO",
        stop={
            "episode_reward_mean": 500
        },
        config={
            "env": "TradingEnv",
            "env_config": {
                "window_size": 24
            },
            "log_level": "DEBUG",
            "framework": "torch",
            "ignore_worker_failures": True,
            "num_workers": 1,
            "num_gpus": 0,
            "clip_rewards": True,
            "lr": 8e-6,
            "lr_schedule": [
                [0, 1e-1],
                [int(1e2), 1e-2],
                [int(1e3), 1e-3],
                [int(1e4), 1e-4],
                [int(1e5), 1e-5],
                [int(1e6), 1e-6],
                [int(1e7), 1e-7]
            ],
            "gamma": 0,
            "observation_filter": "MeanStdFilter",
            "lambda": 0.72,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01
        },
        checkpoint_at_end=True
    )

    # Get checkpoint
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean"
    )
    checkpoint_path = checkpoints[0][0]

    # Restore agent
    agent = ppo.PPOTrainer(
        env="TradingEnv",
        config={
            "env_config": {
                "window_size": 24
            },
            "framework": "torch",
            "log_level": "DEBUG",
            "ignore_worker_failures": True,
            "num_workers": 1,
            "num_gpus": 0,
            "clip_rewards": True,
            "lr": 8e-6,
            "lr_schedule": [
                [0, 1e-1],
                [int(1e2), 1e-2],
                [int(1e3), 1e-3],
                [int(1e4), 1e-4],
                [int(1e5), 1e-5],
                [int(1e6), 1e-6],
                [int(1e7), 1e-7]
            ],
            "gamma": 0,
            "observation_filter": "MeanStdFilter",
            "lambda": 0.72,
            "vf_loss_coeff": 0.5,
            "entropy_coeff": 0.01
        }
    )

    agent.restore(checkpoint_path)

	#Direct Performance and Net Worth Plotting
    performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
    performance.plot()

    portfolio.performance.net_worth.plot()


#Setup Trading Environment
##Create Data Feeds
def create_env(config):
    df = load_csv('data.csv')

    dataset = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna=True)
    display(dataset.head(7))

    #Create Chart Price History Data
    price_history = dataset[['date', 'open', 'high', 'low', 'close', 'volume']]  # chart data
    dataset.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)

    p = Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-BTC")
    binance = Exchange("binance", service=execute_order)(
        p
    )

    cash = Wallet(binance, 10000 * USD)
    asset = Wallet(binance, 0 * BTC)
    portfolio = Portfolio(USD, [
        cash,
        asset
    ])

    with NameSpace("binance"):
        streams = [
            Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns
        ]
    feed = DataFeed(streams)

    #Environment with Multiple Renderers
    chart_renderer = PlotlyTradingChart(
        display=True,
        # None for 100% height.
        height=800,
        save_format="html",
        auto_open_html=True,
    )

    file_logger = FileLogger(
        # omit or None for automatic file name.
        filename="example.log",
        # create a new directory if doesn't exist, None for no directory.
        path="training_logs"
    )
    
    renderer_feed = DataFeed([
        Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
    )

    reward_scheme = default.rewards.RiskAdjustedReturns(window_size=24)
    action_scheme = default.actions.ManagedRiskOrders()

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        window_size=24,
        renderer_feed=renderer_feed,
        renderers=[
        ScreenLogger,
        FileLogger,
        chart_renderer
        ]
    )

    return env


if __name__ == "__main__":
    main()
