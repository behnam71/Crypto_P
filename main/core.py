"""Example of using a custom RNN keras model."""
import argparse
import os
import pandas as pd
from IPython.display import display
import math  

#import tensorflow as tf

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
#import ray.rllib.agents.dqn as dqn
#import ray.rllib.agents.a3c.a2c as a2c
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray import tune

import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.env.default.renderers import PlotlyTradingChart, ScreenLogger
from tensortrade.env.default.actions import ManagedRiskOrders
from tensortrade.env.default.rewards import RiskAdjustedReturns
from tensortrade.env.generic import TradingEnv
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USDT, BTC, DOGE

from talib_indicator import TAlibIndicator

import tickers
import balance
import BinanceData


parser = argparse.ArgumentParser()
parser.add_argument(
    "--alg",
    type=str,
    choices=["PPO", "A2C", "DQN"],
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--symbol",
    type=str,
    choices=["BTC/USDT", "DOGE/USDT"],
    default="BTC/USDT")
parser.add_argument("--num_cpus", type=int, default=2)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument(
    "--stop_iters",
    type=int,
    default=10,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop_timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop_reward",
    type=float,
    default=9000.0,
    help="Reward at which we stop training.")
parser.add_argument(
    "--as_test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")

def data_loading():
    #candles = BinanceData.main()
    candles = pd.read_csv('/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/binance_DOGE.csv', 
                          sep=',', 
                          low_memory=False, 
                          index_col=[0])
    return candles

def start():
    args = parser.parse_args()

    # Declare when training can stop & Never more than 200
    maxIter = args.stop_iters

    # === TRADING ENVIRONMENT CONFIG === 
    # Lookback window for the TradingEnv
    # Increasing this too much can result in errors and overfitting, also increases the duration necessary for training
    # Value needs to be bigger than 1, otherwise it will take nothing in consideration
    window_size = 6

    # 1 meaning he cant lose anything 0 meaning it can lose everything
    # Setting a high value results in quicker training time, but could result in overfitting
    # Needs to be bigger than 0.2 otherwise test environment will not render correctly.
    max_allowed_loss = 0.95

    # === CONFIG FOR AGENT ===
    config = {
        # === ENV Parameters ===
        "env" : "TradingEnv",
        "env_config" : {
            "window_size" : window_size,
            "max_allowed_loss" : max_allowed_loss,
            "train" : not(args.as_test)
        },
        # === RLLib parameters ===
        # https://docs.ray.io/en/master/rllib-training.html#common-parameters
        # === Settings for Rollout Worker processes ===
        # Number of rollout worker actors to create for parallel sampling.
        "num_workers" : args.num_cpus - 3, # Amount of CPU cores - 1


        # === Environment Settings ===
        # Discount factor of the MDP.
        # Lower gamma values will put more weight on short-term gains, whereas higher gamma values will put more weight towards long-term gains. 
        "gamma" : 0, # default = 0.99
        
        # Use GPUs iff "RLLIB_NUM_GPUS" env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        #"num_gpus": 1,
                
        #"lr" : 0.01, # default = 0.00005 && Higher lr fits training model better, but causes overfitting 
        #"clip_rewards": True, 
        #"observation_filter": "MeanStdFilter",
        #"lambda": 0.72,
        #"batch_mode": "complete_episodes",

        # === Debug Settings ===
        "log_level" : "WARN", # "WARN" or "DEBUG" for more info
        "ignore_worker_failures" : True,

        # === Custom Metrics === 
        "callbacks": {"on_episode_end": get_net_worth},

        "model": {
            "custom_model": "rnn",
            "max_seq_len": 20,
            "custom_model_config": {
                "cell_size": 32
            },
        },
        "framework": args.framework,
    }

    # Setup Trading Environment
    ## Create Data Feeds
    def create_env(config):
        symbol = args.symbol
        coin = symbol.split("/")[0]
        print("symbol_Instrument: {}".format(symbol))

        # Use config param to decide which data set to use
        candles = data_loading()
        # Add prefix in case of multiple assets
        data = candles.add_prefix(coin + ":")
        df = data; env_Data = candles
        ta_Data = candles
        p = Stream.source(df[(coin + ':close')].tolist(), dtype="float").rename(("USDT-" + coin))

        # === EXCHANGE ===
        # Commission on Binance is 0.075% on the lowest level, using BNB (https://www.binance.com/en/fee/schedule)
        binance_options = ExchangeOptions(commission=0.001)
        binance = Exchange("binance", service=execute_order, t_signal=config["train"], options=binance_options)(
            p
        )

        # === ORDER MANAGEMENT SYSTEM ===
        if symbol == 'DOGE/USDT':
            symbol_Instrument = DOGE
            #price = tickers.main(symbol)
            min_order_abs = 10
            print("minimum order size: {}".format(str(min_order_abs)))
            if not(config["train"]):
                #usdt_balance, quote_balance = balance.main(coin)
                usdt_balance = 50; quote_balance = 0
            else:
                usdt_balance = 100000; quote_balance = 0
        else:
            symbol_Instrument = BTC
            #price = tickers.main(symbol)
            min_order_abs = 10
            print("minimum order size: {}".format(str(min_order_abs)))
            if not(config["train"]):
                #usdt_balance, quote_balance = balance.main(coin)
                usdt_balance = 50; quote_balance = 0
            else:
                usdt_balance = 100000; quote_balance = 0

        cash = Wallet(binance, usdt_balance * USDT)
        asset = Wallet(binance, quote_balance * symbol_Instrument)
        portfolio = Portfolio(USDT, [
            cash,
            asset
        ])
        
        # === OBSERVER ===
        dataset = pd.DataFrame()
        with open("/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/indicators.txt", "r") as file:
            indicators_list = eval(file.readline())
        TAlib_Indicator = TAlibIndicator(indicators_list)
        dataset = TAlib_Indicator.transform(ta_Data)
        dataset.set_index('date', inplace = True)
        dataset = dataset.add_prefix(coin + ":")
        if config["train"]:
            display(dataset.head(200))
        with NameSpace("binance"):
            streams = [
                Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns
            ]
        # This is everything the agent gets to see, when making decisions
        feed = DataFeed(streams)
        
        # Compiles all the given stream together
        feed.compile()

        # === REWARDSCHEME ===
        # RiskAdjustedReturns rewards depends on return_algorithm and its parameters. SimpleProfit() or RiskAdjustedReturns() or PBR()
        #reward_scheme = SimpleProfit(window_size=config["window_size"])
        #reward_scheme = RiskAdjustedReturns(return_algorithm='sortino')#, risk_free_rate=0, target_returns=0)
        reward_scheme = RiskAdjustedReturns(return_algorithm='sharpe',
                                            risk_free_rate=0,
                                            target_returns=0,
                                            window_size=config["window_size"]
                                            )

        # === ACTIONSCHEME ===
        # SimpleOrders() or ManagedRiskOrders() or BSH()
        action_scheme = ManagedRiskOrders(
            stop = [0.02],
            take = [0.03],
            durations=[100],
            trade_sizes=10,
            min_order_abs=min_order_abs
        )

        # === RENDERER ===
        chart_renderer = PlotlyTradingChart(
            display=True, # show the chart on screen (default)
            height=800, # affects both displayed and saved file height. None for 100% height.
            save_format="html", # save the chart to an HTML file
            auto_open_html=True # open the saved HTML chart in a new browser tab
        )
        # Uses the OHCLV data passed to envData
        renderer_feed = DataFeed([
            Stream.source(env_Data[c].tolist(), dtype="float").rename(c) for c in env_Data]
        )

        # === RESULT === 
        env = default.create(
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            feed=feed,
            window_size=config["window_size"], # part of OBSERVER
            max_allowed_loss=config["max_allowed_loss"], # STOPPER
            enable_logger=True,
            t_signal=config["train"],
            renderer_feed=renderer_feed,
            renderer= PlotlyTradingChart(), # PositionChangeChart()
            renderers=[
                ScreenLogger,
                chart_renderer
            ]
        )

        return env

    register_env("TradingEnv", create_env)

    # === Scheduler ===
    # Currenlty not in use
    # https://docs.ray.io/en/master/tune/api_docs/schedulers.html
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

    if not ray.is_initialized():
       ray.init(num_cpus=args.num_cpus or None, local_mode=True)
       #ray.init(num_gpus=1) # Skip or set to ignore if already called

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if args.framework == "torch" else RNNModel)
    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    if not(args.as_test):
        analysis = tune.run(
            args.alg,
            # https://docs.ray.io/en/master/tune/api_docs/stoppers.html
            stop = {
                "training_iteration": maxIter,
                "episode_reward_mean": args.stop_reward,
                #"timesteps_total": args.stop_timesteps
            },
            config=config,
            checkpoint_at_end=True,
            checkpoint_freq=1, # Necesasry to declare, in combination with Stopper
            checkpoint_score_attr="episode_reward_mean",
            #resume=True,
            #restore="/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results/PPO",
            local_dir="/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/crypto-v1/ray_results",
            #scheduler=asha_scheduler,
            #max_failures=5
        )

        #if args.as_test:
            #check_learning_achieved(analysis, args.stop_reward)

    else:
        ###########################################
        # === ANALYSIS FOR TESTING ===
        # https://docs.ray.io/en/master/tune/api_docs/analysis.html
        # Get checkpoint based on highest episode_reward_mean
        from ray.tune import Analysis
        analysis = Analysis("/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results/PPO")
        checkpoint_path = analysis.get_best_checkpoint(
            trial="/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results/PPO/PPO_TradingEnv_77cab_00000_0_2021-08-14_08-13-12",
            metric="episode_reward_mean",
            mode="max"
        ) 
        print("Checkpoint Path at: {}".format(str(checkpoint_path)))

        # === ALGORITHM SELECTION ===   
        # Get the correct trainer for the algorithm
        if (args.alg == "PPO"):
            algTr = ppo.PPOTrainer
        if (args.alg == "DQN"):
            algTr = dqn.DQNTrainer
        if (args.alg == "A2C"):
            algTr = a2c.A2CTrainer

        # === CREATE THE AGENT === 
        agent = algTr(
            env="TradingEnv", config=config,
        )
        # Restore agent using best episode reward mean
        agent.restore(checkpoint_path)

        # Instantiate the testing environment
        # Must have same settings for window_size and max_allowed_loss as the training env
        test_env = create_env({
            "window_size": window_size,
            "max_allowed_loss": max_allowed_loss,
            "train": False
        })

        # === Render the environments (online) ===
        render_env(test_env, agent)

    if ray.is_initialized():
        ray.shutdown()


def render_env(env, agent):
    # Run until done == True
    done = False
    obs = env.reset()
    state = agent.get_policy().get_initial_state()

    print("Start Interaction ...")
    while not done:
        action, state, _ = agent.compute_action(
            obs,
            state=state
        )
        print("\nSelected Action: {}".format(str(action)))
        obs = env.step(action)

    # Render the test environment
    #env.render()

# === CALLBACK ===
def get_net_worth(info):
    # info is a dict containing: env, policy and info["episode"] is an evaluation episode
    episode = info["episode"]
    episode.custom_metrics["net_worth"] = episode.last_info_for()["net_worth"]


if __name__ == "__main__":
    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    #tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    start()

    # tensorboard --logdir=/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/ray_results/PPO
    # python core.py --alg PPO --symbol "DOGE/USDT" --stop_reward 4.05549e+6 --num_cpus 4 --framework torch --stop_iters 200
    # python core.py --alg PPO --symbol "DOGE/USDT" --num_cpus 4 --framework torch --as_test
