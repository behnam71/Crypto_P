"""Example of using a custom RNN keras model."""
import argparse
import os

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved

import pandas as pd
import ta
from IPython.display import display

from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger, ScreenLogger
import tensortrade.env.default as default


parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=100,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=90.0,
    help="Reward at which we stop training.")

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
    args = parser.parse_args()

    ray.init(num_cpus=args.num_cpus or None)

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if args.framework == "torch" else RNNModel)

    register_env("TradingEnv", create_env)

    analysis = tune.run(
        "PPO",
        stop={
            "training_iteration": args.stop_iters,
            "timesteps_total": args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,        
        },

        config = {
            "env": 'TradingEnv',
            "env_config": {
                "window_size": 24,
                "repeat_delay": 2,
            },
            "log_level": "DEBUG",
            "gamma": 0.9,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 1,
            "entropy_coeff": 0.001,
            "num_sgd_iter": 5,
            "vf_loss_coeff": 1e-5,
            "model": {
                "custom_model": "rnn",
                "max_seq_len": 20,
                "custom_model_config": {
                    "cell_size": 32,
                },
            },
            "framework": args.framework,
        },
        checkpoint_at_end=True
    )

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()

    """
    # Restore agent
    import ray.rllib.agents.ppo as ppo

    # Get checkpoint
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric="episode_reward_mean", , mode="min"),
        metric="episode_reward_mean"
    )
    checkpoint_path = checkpoints[0][0]

    agent = ppo.PPOTrainer(
        env="TradingEnv",
        config = {
            "env": args.env,
            "env_config": {
                "window_size": 24
                "repeat_delay": 2,
            },
            "gamma": 0.9,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "num_workers": 1,
            "num_envs_per_worker": 20,
            "entropy_coeff": 0.001,
            "num_sgd_iter": 5,
            "vf_loss_coeff": 1e-5,
            "model": {
                "custom_model": "rnn",
                "max_seq_len": 20,
                "custom_model_config": {
                    "cell_size": 32,
                },
            },
            "framework": args.framework,
        },
    )

    agent.restore(checkpoint_path)
    """

	#Direct Performance and Net Worth Plotting
    performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
    performance.plot()

    portfolio.performance.net_worth.plot()


#Setup Trading Environment
##Create Data Feeds
def create_env(config):
    df = load_csv('data.csv')
    """
    dataset = ta.add_all_ta_features(df, 'open', 'high', 'low', 'close', 'volume', fillna=True)
    """
    dataset = df
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
