import gymnasium as gym
import numpy as np
from pathlib import Path
from BlackJackObservationWrapper import BlackjackContinuousWrapper
from OneHotObservationWrapper import OneHotObservationWrapper
from dqn.agent import DQNAgent
from dqn.configs import HYPERPARAMS
from dqn.configs import PREF_WEIGHTS
# Create necessary directories if they don't exist
DQN_LOGS_DIR = Path(__file__).parent.joinpath('dqn/logs')
DQN_LOGS_DIR.mkdir(parents=True, exist_ok=True)
(DQN_LOGS_DIR / 'episode').mkdir(exist_ok=True)
(DQN_LOGS_DIR / 'tamer').mkdir(exist_ok=True)
(DQN_LOGS_DIR / 'gifs').mkdir(exist_ok=True)


DQN_MODELS_DIR = Path(__file__).parent.joinpath('dqn/saved_models')
DQN_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # lake_env = OneHotObservationWrapper(gym.make(
    #     'FrozenLake-v1',
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    #     reward_schedule=(100, -50, -1),
    #     render_mode="rgb_array"))
    #lake_action_map = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}

    # regular LunarLander env 
    # agent = DQNAgent(
    #     **HYPERPARAMS['LunarLander-v3'],
    #     num_episodes=500,
    #     ts_len=0,
    #     q_model_to_load=None,
    #     h_model_to_load=None,
    #     gif_name="jan_dqn_ll_500eps.gif",
    #     render=True,
    #     tamer=False)

    agent = DQNAgent(
        **HYPERPARAMS['LunarLander-v3'],
        num_episodes=100,
        ts_len=0,
        q_model_to_load=None,
        h_model_to_load=None,
        gif_name="jan_pref_learning_dqn_lunarlander_100eps.gif",
        render=True,
        tamer=False,
        PREF_TRAJECTORY= True,
        PREF_PARAMS= PREF_WEIGHTS['LunarLander-v3'])

    # this already saves the gif after training 
    agent.train(
        name="DQN LunarLander 100 eps Preference Learning",
        q_model_file_to_save="pref_q_dqntamer_ll_100eps.pth",
        h_model_file_to_save="pref_h_dqntamer_ll_100eps.pth",
        eval=True,
        eval_interval=20
    )
    #agent.play(n_episodes=1, render=True, save_gif=True,
    #            gif_name="ll_dqn_500eps.gif")


if __name__ == '__main__':
    main()
