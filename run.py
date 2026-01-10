"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gymnasium as gym
import numpy as np

from tamer.agent import TamerRL
from pathlib import Path
from tamer.configs import HYPERPARAMS
from tamer.configs import PREF_WEIGHTS
from gymnasium import spaces


MODELS_DIR = Path(__file__).parent.joinpath('tamer/saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('tamer/logs')

# Create necessary directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
(LOGS_DIR / 'episode').mkdir(exist_ok=True)
(LOGS_DIR / 'tamer').mkdir(exist_ok=True)
(LOGS_DIR / 'gifs').mkdir(exist_ok=True)



async def main():
    # Experiment params
    num_episodes = 100
    control_sharing = False
    tamer_timestep_length = 0

    # Domain specific params
    # MountainCar
    mc_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    mc_params = HYPERPARAMS["MountainCar-v0"]
    mc_params["trace_scaling"] = 2 if control_sharing else 100

    MC_PREF_PARAMS = PREF_WEIGHTS["MountainCar-v0"]

    mc_agent_config = {
        **mc_params,
        "env": mc_env,
        "num_episodes": num_episodes,
        "control_sharing": control_sharing,
        "ts_len": tamer_timestep_length,
        "logs_dir": LOGS_DIR,
        "models_dir": MODELS_DIR,
        "q_model_to_load": None,
        "h_model_to_load": None,
        "PREF_TRAJECTORY": True,
        "PREF_PARAMS": MC_PREF_PARAMS
    }

    # CartPole
    cp_env = gym.make("CartPole-v1", render_mode="rgb_array")
    cp_params = HYPERPARAMS["CartPole-v1"]
    cp_params["trace_scaling"] = 1 if control_sharing else 200

    CP_PREF_PARAMS = PREF_WEIGHTS["CartPole-v1"]

    cp_agent_config = {
        **cp_params,
        "env": cp_env,
        "num_episodes": num_episodes,
        "control_sharing": control_sharing,
        "ts_len": tamer_timestep_length,
        "logs_dir": LOGS_DIR,
        "models_dir": MODELS_DIR,
        "q_model_to_load": None,
        "h_model_to_load": None,
        "PREF_TRAJECTORY": True,
        "PREF_PARAMS": CP_PREF_PARAMS
    }

    # Lunar Lander
    ll_env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                      enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
    ll_params = PREF_WEIGHTS["LunarLander-v3"]
    ll_agent_config = {
        **ll_params,
        "env": ll_env,
        "num_episodes": num_episodes,
        "control_sharing": control_sharing,
        "ts_len": tamer_timestep_length,
        "logs_dir": LOGS_DIR,
        "models_dir": MODELS_DIR,
        "q_model_to_load": None,
        "h_model_to_load": None
    }

    agent = TamerRL(**cp_agent_config)
    #agent = TamerRL(**mc_agent_config)

    await agent.train(model_file_to_save="100eps_cp_pref_learning_200ts.p", eval=True, eval_interval=20)
    agent.play(n_episodes=3, render=True, save_gif=True)
    # agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())
