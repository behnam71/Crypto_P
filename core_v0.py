"""Example of using a custom RNN keras model."""
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
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
from ray.tune.stopper import ExperimentPlateauStopper
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray import tune

from tensortrade.environments import TradingEnvironment
from tensortrade.rewards import RiskAdjustedReturns
from tensortrade.actions import ManagedRiskOrders
from tensortrade.instruments import BTC, USD, Quantity
from tensortrade.wallets import Wallet, Portfolio
from tensortrade.exchanges.simulated import SimulatedExchange

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
    "--online",
    type=bool,
    default=False,
    help="Testing online or offline."
    )
parser.add_argument(
    "--window_size",
    type=int,
    default=1,
    help="Testing online or offline."
    )

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
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        #"num_gpus": 1,

        #"train_batch_size": 1000,

        # === Debug Settings ===
        "log_level": "WARN", # "WARN" or "DEBUG" for more info
        "ignore_worker_failures": True,

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
        candles = pd.read_csv('/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/data.csv', 
                              low_memory=False, 
                              index_col=[0])

        # === EXCHANGE ===
        # Commission on Binance is 0.075% on the lowest level, using BNB (https://www.binance.com/en/fee/schedule)
        binance = SimulatedExchange(data_frame=candles, 
                                    price_column="close",
                                    randomize_time_slices=True, 
                                    commission=0.0075,
                                    min_trade_price=10.0)

        # === ORDER MANAGEMENT SYSTEM === 
        # Start with 100.000 usd and 0 assets
        cash = Wallet(binance, Quantity(USD, 10000))
        asset = Wallet(binance, Quantity(BTC, 0))

        portfolio = Portfolio(USD, [
            cash,
            asset
        ])

        # === REWARDSCHEME === 
        reward_scheme = reward_scheme = RiskAdjustedReturns(
            return_algorithm='sharpe', 
            risk_free_rate=0, 
            target_returns=0, 
        )     

        # === ACTIONSCHEME ===
        action_scheme = ManagedRiskOrders(
            pairs=[USD/BTC],
            stop_loss_percentages = [0.02], 
            take_profit_percentages = [0.03], 
            trade_sizes=2
        )

        # === RESULT === 
        environment = TradingEnvironment(
            exchange=binance,
            portfolio=portfolio,
            observation_highs=1.0e+10,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            window_size=config["window_size"], # part of OBSERVER
            observe_wallets=[USD, BTC]
        )
        return environment

    register_env("TradingEnv", create_env)

    if not ray.is_initialized():
        ray.init(num_cpus=args.num_cpus or None)
        #ray.init(num_gpus=1) # Skip or set to ignore if already called

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if args.framework == "torch" else RNNModel)
    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    if not(args.online):
        analysis = tune.run(
            args.alg,
            # https://docs.ray.io/en/master/tune/api_docs/stoppers.html
            stop={"training_iteration": maxIter,
            #"timesteps_total": args.stop_timesteps, "episode_reward_mean": args.stop_reward,
            },
            config=config,
            checkpoint_at_end=True,
            metric="episode_reward_mean",
            mode="max", 
            checkpoint_freq=1, # Necesasry to declare, in combination with Stopper
            checkpoint_score_attr="episode_reward_mean",
            local_dir= "~/ray_results"
            #restore="~/ray_results/PPO",
            #resume=True,
            #max_failures=100,

        )

    else:
        #check_learning_achieved(analysis, args.stop_reward)

        ###########################################
        # === ANALYSIS FOR TESTING ===
        # https://docs.ray.io/en/master/tune/api_docs/analysis.html
        # Get checkpoint based on highest episode_reward_mean
        from ray.tune import Analysis
        analysis = Analysis("~/ray_results/PPO")
        checkpoint_path = analysis.get_best_checkpoint(
            trial="~/ray_results/PPO/PPO_TradingEnv_f15df_00000_0_2021-06-22_08-29-21", 
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
        })

        render_env(test_env, agent)

    ray.shutdown()


def render_env(env, agent):
    # Run until done == True
    done = False
    obs = env.reset()

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
        print("Next Observer:"); print(obs)
        print("Selected Action: {}".format(str(action)))
        print("Reward: {}".format(str(reward)))
        print("Total Reward: {}".format(str(total_reward)))
        sleep(2)
    
    # Render the test environment
    env.render()


if __name__ == "__main__":
    args = parser.parse_args()

    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    #tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    if args.online:
        # creating processes
        process = multiprocessing.Process(target=fetchData, args=('1m', [args.c_Instrument + "/USDT"], args.window_size,))
        process.start()
        while True:
            modTimesinceEpocORG = os.path.getmtime("data.csv")
            modificationTimeORG = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpocORG))
            temp = modificationTimeORG
            modificationTime = temp
            while modificationTime == temp:
                temp = modificationTime
                modTimesinceEpoc = os.path.getmtime("data.csv")
                modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
            main_process(args)
    else:
        main_process(args)


    # tensorboard --logdir=C:\Users\Stephan\ray_results\PPO

    # python core.py --alg PPO --c_Instrument BTC --num_cpus 3 --framework torch --stop_iters 120 --window_size 20
    # python core.py --alg PPO --c_Instrument BTC --num_cpus 3 --framework torch --stop_iters 120 --online --window_size 20
