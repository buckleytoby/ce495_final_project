import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Assuming your environment setup is in a file named low_dof_env_factory.py
# or defined earlier in your script
from low_dof_gym_env import LowDOFRotateGymEnv

from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
        
from stable_baselines3.common.vec_env import VecMonitor

from stable_baselines3.common.vec_env import VecEnv
        
def main():
    # 1. Create the environment
    # DrakeGymEnv usually handles the observation and action spaces based on 
    # the input/output ports you defined in your Diagram.
    env: VecEnv = LowDOFRotateGymEnv()
    
    # wrap with Monitor to log episode rewards and lengths
    log_dir = "./puck_logs/"
    env = VecMonitor(env)
    

    # 3. Initialize the PPO Agent
    # We use MlpPolicy because the state is likely a small vector
    # (joint position, velocity, and maybe contact wrenches).
        
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=50,      # Experience collected before an update
        batch_size=64,     # Size of SGD minibatches
        n_epochs=10,       # Optimization epochs per update
        gamma=0.99,        # Discount factor
        verbose=1,
        tensorboard_log=log_dir,
    )
    
    model.learn(total_timesteps=100_000, progress_bar=True, log_interval=1)

if __name__ == "__main__":
    main()