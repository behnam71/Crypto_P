"""
Example of using an RL agent (default: PPO) with an AttentionNet model,
which is useful for environments where state is important but not explicitly
part of the observations.
For example, in the "repeat after me" environment (default here), the agent
needs to repeat an observation from n timesteps before.
AttentionNet keeps state of previous observations and uses transformers to
learn a policy that successfully repeats previous observations.
Without attention, the RL agent only "sees" the last observation, not the one
n timesteps ago and cannot learn to repeat this previous observation.
AttentionNet paper: https://arxiv.org/abs/1506.07704
This example script also shows how to train and test a PPO agent with an
AttentionNet model manually, i.e., without using Tune.
---
Run this example with defaults (using Tune and AttentionNet on the "repeat
after me" environment):
$ python attention_net.py
Then run again without attention:
$ python attention_net.py --no-attention
Compare the learning curve on TensorBoard:
$ cd ~/ray-results/; tensorboard --logdir .
There will be a huge difference between the version with and without attention!
Other options for running this example:
$ python attention_net.py --help
"""
import argparse
import os

import numpy as np
import pandas as pd
import math  
from IPython.display import display

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.agents import dqn
from ray.rllib.examples.env.look_and_push import LookAndPush, OneHot
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.examples.env.repeat_initial_obs_env import RepeatInitialObsEnv
from ray.rllib.examples.env.stateless_cartpole import StatelessCartPole
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune import registry
from ray.tune.logger import pretty_print
#import tensorflow as tf

#import ray.rllib.agents.dqn as dqn
#from ray.tune.schedulers import ASHAScheduler

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

tf1, tf, tfv = try_import_tf()


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # example-specific args
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Do NOT use attention. For comparison: The agent will not learn."
    )

    # general args
    parser.add_argument(
        "--run", default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=3)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="tf",
        help="The DL framework specifier."
    )
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=200,
        help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=500000,
        help="Number of timesteps to train."
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=80.0,
        help="Reward at which we stop training."
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters."
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. Here,"
        "there is no TensorBoard support."
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging."
    )

    parser.add_argument(
        "--symbol",
        type=str,
        choices=["BTC/USDT", "DOGE/USDT"],
        default="BTC/USDT"
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    def data_loading():
        #candles = BinanceData.main()
        candles = pd.read_csv('/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/binance_DOGE.csv', 
                              sep=',', 
                              low_memory=False,
                              index_col=[0])
        return candles


    def render_env(env, agent):
        obs = env.reset()
        done = False
      
        # start with all zeros as state
        num_transformers = config["model"][
            "attention_num_transformer_units"
        ]
        init_state = state = [
            np.zeros([100, 32], np.float32)
            for _ in range(num_transformers)
        ]

        # run one iteration until done
        print(f"TradingEnv with {config['env_config']}")
        while not done:
            action, state_out, _ = trainer.compute_single_action(
                obs, state
            )
            next_obs = env.step(action)

            print(f"Obs: {obs}, Action: {action}")
            obs = next_obs

            state = [
                np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
                for i in range(num_transformers)
            ]

        # Render the test environment
        #env.render()

    # Setup Trading Environment
    ## Create Data Feeds
    def create_env(config):
        symbol = args.symbol
        coin = symbol.split("/")[0]
        print("symbol_Instrument: {}".format(symbol))

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
            trade_sizes=4,
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


    if not ray.is_initialized():
       ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)
       #ray.init(num_gpus=1)

    # register custom environments
    registry.register_env("TradingEnv", create_env)

    window_size = 48

    # main part: RLlib config with AttentionNet model
    config = {
        "env": "TradingEnv",
        # This env_config is only used for the RepeatAfterMeEnv env.
        "env_config": {
            "window_size" : window_size,
            "train" : not(args.as_test)
        },

        # === RLLib parameters ===
        # https://docs.ray.io/en/master/rllib-training.html#common-parameters
        # === Settings for Rollout Worker processes ===
        # Number of rollout worker actors to create for parallel sampling.
        "num_workers" : args.num_cpus - 1, # Amount of CPU cores - 1
        "num_envs_per_worker": 1,

        "gamma": 0.99,

        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
        #"num_gpus": 1,

        "train_batch_size": 48,
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        #"timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    if not(args.as_test):
        # training loop
        if args.no_tune:
            # manual training loop using PPO and manually keeping track of state
            if args.run != "PPO":
                raise ValueError("Only support --run PPO with --no-tune.")
            ppo_config = ppo.DEFAULT_CONFIG.copy()
            ppo_config.update(config)
            trainer = ppo.PPOTrainer(config=ppo_config, env=TradingEnv)
            # run manual training loop and print results after each iteration
            for i in range(args.stop_iters):
                result = trainer.train()
                pretty_print(result)
                
                checkpoint = train_agent.save(tune.get_trial_dir())
                tune.report(**train_results)

                # stop training if the target train steps or reward are reached
                if result["timesteps_total"] >= args.stop_timesteps or \
                        result["episode_reward_mean"] >= args.stop_reward:
                    break
            train_agent.stop()

            # run manual test loop.
            print("Finished training. Running manual test/inference loop.")

        else:
            # run with Tune for auto env and trainer creation and TensorBoard
            results = tune.run(
                args.run,
                # https://docs.ray.io/en/master/tune/api_docs/stoppers.html
                stop=stop,
                config=config,
                checkpoint_at_end=True,
                checkpoint_freq=1, # Necesasry to declare, in combination with Stopper
                checkpoint_score_attr="episode_reward_mean",
                #scheduler=asha_scheduler,
                local_dir="/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results",
            )

            #print("Checking if learning goals were achieved")
            #check_learning_achieved(results, args.stop_reward)

    else:
        ###########################################
        # === ANALYSIS FOR TESTING ===
        # https://docs.ray.io/en/master/tune/api_docs/analysis.html
        # Get checkpoint based on highest episode_reward_mean
        from ray.tune import Analysis
        analysis = Analysis("/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results/DQN")
        checkpoint_path = analysis.get_best_checkpoint(
            trial="/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results/DQN/DQN_TradingEnv_a4fd9_00000_0_2021-08-18_09-57-06",
            metric="episode_reward_mean",
            mode="max"
        ) 
        print("Checkpoint Path at: {}".format(str(checkpoint_path)))

        # === ALGORITHM SELECTION ===   
        # Get the correct trainer for the algorithm
        if (args.run == "PPO"):
            algTr = ppo.PPOTrainer
        if (args.run == "DQN"):
            algTr = dqn.DQNTrainer

        # === CREATE THE AGENT === 
        agent = algTr(config=config, env="TradingEnv")
        
        # Restore agent using best episode reward mean
        agent.restore(checkpoint_path)

        # Instantiate the testing environment
        # Must have same settings for window_size and max_allowed_loss as the training env
        test_env = create_env({"window_size": window_size, "train": False})

        # === Render the environments (online) ===
        render_env(test_env, agent)

    if ray.is_initialized():
        ray.shutdown()



#tensorboard --logdir=/mnt/c/Users/BEHNAMH721AS.RN/OneDrive/Desktop/crypto_v2/ray_results/DQN
#python crypto_v2/core_v2.py --run DQN --symbol "DOGE/USDT" --num-cpus 4 --framework torch --stop-reward 4.0e+7 --local-mode --stop-iters 300
#python crypto_v2/core_v2.py --run DQN --symbol "DOGE/USDT" --num-cpus 2 --framework torch --local-mode --as-test