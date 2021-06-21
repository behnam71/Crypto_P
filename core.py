"""Example of using a custom RNN keras model."""
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from time import sleep
from pprint import pprint

import tensorflow as tf

import ray
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.a3c.a2c as a2c
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import ExperimentPlateauStopper
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray import tune

import tensortrade.env.default as default
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger, ScreenLogger
from tensortrade.env.default.actions import TensorTradeActionScheme, ManagedRiskOrders
from tensortrade.env.default.rewards import TensorTradeRewardScheme, SimpleProfit, RiskAdjustedReturns
from tensortrade.env.generic import ActionScheme, TradingEnv, Renderer
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair, Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)

import multiprocessing
from ta import add_all_ta_features

from continuously_Data import fetchData


parser = argparse.ArgumentParser()
parser.add_argument(
    "--alg",
    type=str,
    choices=["PPO", "A2C", "DQN"],
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--c_Instrument",
    type=str,
    choices=["BTC", "ETH", "DOGE"],
    default="BTC"
    )
parser.add_argument(
    "--num_cpus", 
    type=int, 
    default=3
    )
parser.add_argument(
    "--stop_iters",
    type=int,
    default=120,
    help="Number of iterations to train."
    )
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier."
    )
parser.add_argument(
    "--stop_reward",
    type=float,
    default=9000.0,
    help="Reward at which we stop training."
    )
parser.add_argument(
    "--stop_timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train."
    )
parser.add_argument(
    "--as_test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters."
    )
parser.add_argument(
    "--online",
    type=bool,
    default=False,
    help="Testing online or offline."
    )
parser.add_argument(
    "--window_size",
    type=int,
    help="Testing online or offline."
    )

def data_loading(args):
    ttp = 0.9
    if args.online == True:
        ttp = 1
    
    dataset = pd.read_csv('./crypto/data.csv', low_memory=False, index_col=[0])
    candles = dataset[['date', 'open', 'high', 'low', 'close', 'volume']] # chart data
    # Divide the data in test (last 20%) and training (first 80%)
    data_End = (int)(len(candles)*ttp)
    return dataset, candles, data_End


def main_process(args):
    # Declare when training can stop & Never more than 200
    maxIter = args.stop_iters

    # === TRADING ENVIRONMENT CONFIG === 
    # Lookback window for the TradingEnv
    # Increasing this too much can result in errors and overfitting, also increases the duration necessary for training
    # Value needs to be bigger than 1, otherwise it will take nothing in consideration
    window_size = args.window_size

    # 1 meaning he cant lose anything 0 meaning it can lose everything
    # Setting a high value results in quicker training time, but could result in overfitting
    # Needs to be bigger than 0.2 otherwise test environment will not render correctly.
    max_allowed_loss = 0.95

    # === CONFIG FOR AGENT ===
    config = {
        # === ENV Parameters ===
        "env": "TradingEnv",
        "env_config": {
            "window_size": window_size,
            "max_allowed_loss": max_allowed_loss,
            "train": args.as_test,
        },
        # === RLLib parameters ===
        # https://docs.ray.io/en/master/rllib-training.html#common-parameters
        # === Settings for Rollout Worker processes ===
        # Number of rollout worker actors to create for parallel sampling.
        "num_workers": 1, # Amount of CPU cores - 1

        # === Environment Settings ===
        # Discount factor of the MDP.
        # Lower gamma values will put more weight on short-term gains, whereas higher gamma values will put more weight towards long-term gains. 
        "gamma": 0.7, # default = 0.99

        # Use GPUs iff "RLLIB_NUM_GPUS" env var set to > 0.
        #"num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_gpus": 1,

        #"train_batch_size": 1000,

        # === Debug Settings ===
        "log_level": "WARN", # "WARN" or "DEBUG" for more info
        "ignore_worker_failures": True,

        # === Custom Metrics === 
        "callbacks": {"on_episode_end": get_net_worth},

        # === rnn model parameters ===
        #"num_envs_per_worker": 1,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 5,
        "vf_loss_coeff": 1e-5,
        "model": {
            "custom_model": "rnn",
            "max_seq_len": 32,
            "custom_model_config": {
                "cell_size": 24,
            },
        },
        "framework": args.framework,
    }

    # Setup Trading Environment
    ## Create Data Feeds
    def create_env(config):
        coin = "BTC"
        coinInstrument = BTC
        dataset, candles, data_End = data_loading(args)
        # Add prefix in case of multiple assets
        data = candles.add_prefix(coin + ":")

        # Use config param to decide which data set to use
        if config["train"] == True:
            df = data[:-data_End]; env_Data = candles[:-data_End]
            ta_Data = data[:-data_End]
        else:
        	df = data[-data_End:]; env_Data = candles[-data_End:]
        	ta_Data = data[-data_End:]

        # === OBSERVER ===
        p = Stream.source(df[(coin + ':close')].tolist(), dtype="float").rename(("USD-" + coin))

        # === EXCHANGE ===
        # Commission on Binance is 0.075% on the lowest level, using BNB (https://www.binance.com/en/fee/schedule)
        binance_options = ExchangeOptions(commission=0.0075, min_trade_price=10.0)
        binance = Exchange("binance", service=execute_order, options=binance_options)(
            p
        )

        # === ORDER MANAGEMENT SYSTEM === 
        # Start with 100.000 usd and 0 assets
        cash = Wallet(binance, 100000 * USD)
        asset = Wallet(binance, 0 * coinInstrument)
        portfolio = Portfolio(USD, [
            cash,
            asset
        ])

        """
        # === OBSERVER ===
        if args.online == True:
            dataset = pd.DataFrame()
            dataset = ta.add_all_ta_features(ta_Data, 'open', 'high', 'low', 'close', 'volume', fillna=True)
            dataset = dataset.add_prefix(coin + ":")
        """
        dataset.set_index('date', inplace = True)
        dataset = dataset.add_prefix(coin + ":")
        display(dataset.head(7))

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
        reward_scheme = reward_scheme = RiskAdjustedReturns(return_algorithm='sharpe', 
        	                                                risk_free_rate=0, 
        	                                                target_returns=0, 
        	                                                window_size=config["window_size"]
        	                                                )     

        # === ACTIONSCHEME ===
        # SimpleOrders() or ManagedRiskOrders() or BSH()
        action_scheme = ManagedRiskOrders(stop = [0.02], 
        	                              take = [0.03], 
        	                              durations=[100], 
        	                              trade_sizes=2
        	                              )

        # === RENDERER ===
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
            renderer_feed=renderer_feed,
            renderer= PlotlyTradingChart(), # PositionChangeChart()
            window_size=config["window_size"], # part of OBSERVER
            max_allowed_loss=config["max_allowed_loss"], # STOPPER
            renderers=[
                ScreenLogger,
                FileLogger,
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
        #ray.init(num_cpus=args.num_cpus or None, local_mode=True)
        ray.init(num_gpus=1) # Skip or set to ignore if already called

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if args.framework == "torch" else RNNModel)
    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    if not(args.as_test):
        analysis = tune.run(
            args.alg,
            # https://docs.ray.io/en/master/tune/api_docs/stoppers.html
            #stop=ExperimentPlateauStopper(metric="episode_reward_mean", std=0.1, top=10, mode="max", patience=0),
            stop={"training_iteration": maxIter,
                  #"timesteps_total": args.stop_timesteps, "episode_reward_mean": args.stop_reward,
            },
            #stop={"episode_len_mean": (len(data) - dataEnd) - 1},
            config=config,
            checkpoint_at_end=True,
            metric="episode_reward_mean",
            mode="max", 
            checkpoint_freq=1, # Necesasry to declare, in combination with Stopper
            checkpoint_score_attr="episode_reward_mean",
            #restore="~/ray_results/PPO",
            #resume=True,
            #scheduler=asha_scheduler,
            #max_failures=5,
        )
        #if args.as_test:
            #check_learning_achieved(analysis, args.stop_reward)

    else:
        ###########################################
        # === ANALYSIS FOR TESTING ===
        # https://docs.ray.io/en/master/tune/api_docs/analysis.html
        # Get checkpoint based on highest episode_reward_mean
        from ray.tune import Analysis
        analysis = Analysis("~/ray_results/PPO")
        checkpoint_path = analysis.get_best_checkpoint(
            trial="~/ray_results/PPO/PPO_TradingEnv_78abe_00000_0_2021-06-17_19-57-05", 
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
            "window_size": config["window_size"],
            "max_allowed_loss": config["max_allowed_loss"],
            "train": False
        })

        render_env(test_env, agent)

    ray.shutdown()


def render_env(env, agent):
    # Run until done == True
    done = False
    obs = env.reset()
    # Start with initial capital
    networth = [0]

    _prev_action = np.zeros_like(env.action_space.sample())
    _prev_reward = 0
    info = {}
    state = agent.get_policy().get_initial_state()
    total_reward = 0
    print("Start Interaction ...")
    while not done:
        action, state, fetch = agent.compute_action(
            obs,
            state=state,
            prev_action=_prev_action,
            prev_reward=_prev_reward,
            info=info
        )
        obs, reward, done, info = env.step(action)
        total_reward = total_reward + reward
        _prev_reward = reward
        _prev_action = action
        networth.append(info['net_worth'])
        print("Next Observer:"); print(obs)
        print("Selected Action: {}".format(str(action)))
        print("Reward: {}".format(str(reward)))
        print("Total Reward: {}".format(str(total_reward)))
        print("NetWorth: {}".format(str(round(info['net_worth'], 2))))
        print("Counter: {}\n\n".format(str(h_counter)))
        sleep(2)
    
    # Render the test environment
    env.render()

    print("NetWorth Ploting:")
    # Direct Performance and Net Worth Plotting
    #performance = pd.DataFrame.from_dict(env.action_scheme.portfolio.performance, orient='index')
    #performance.plot()


# === CALLBACK ===
def get_net_worth(info):
    # info is a dict containing: env, policy and info["episode"] is an evaluation episode
    episode = info["episode"]
    episode.custom_metrics["net_worth"] = episode.last_info_for()["net_worth"]


if __name__ == "__main__":
    args = parser.parse_args()

    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    if args.online == True:
        # creating processes
        process1 = multiprocessing.Process(target=fetchData, args=('1m', [args.c_Instrument + "/USDT"], args.window_size,))
        # starting process 1
        process1.start()
        while True:
            modTimesinceEpocORG = os.path.getmtime("data.csv")
            modificationTimeORG = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpocORG))
            temp = modificationTimeORG
            while modificationTime == temp:
                temp = modificationTime
                modTimesinceEpoc = os.path.getmtime("data.csv")
                modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
            main_process(args)
    else:
        main_process(args)


    # tensorboard --logdir=C:\Users\Stephan\ray_results\PPO
    # python core.py --alg PPO --c_Instrument BTC --num_cpus 3 --framework torch --stop_iters 120 --window_size 20
    # python core.py --alg PPO --c_Instrument BTC --num_cpus 3 --framework torch --stop_iters 120 --as_test --window_size 20

    # python core.py --alg PPO --c_Instrument BTC --num_cpus 3 --framework torch --stop_iters 120 --as_test --online True --window_size 20
