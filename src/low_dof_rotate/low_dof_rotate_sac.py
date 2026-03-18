import os
import gymnasium as gym
from stable_baselines3 import SAC
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

import time

import torch as th

class InferenceTimeCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.inference_times = []

    def _on_step(self) -> bool:
        # We access the most recent observation from the environment
        obs = self.locals["new_obs"]
        
        # Measure only the prediction/inference part
        start_time = time.perf_counter()
        with th.no_grad():
            self.model.policy.predict(obs, deterministic=True)
        end_time = time.perf_counter()
        
        inf_time_ms = (end_time - start_time) * 1000
        self.logger.record("time/inference_time_ms", inf_time_ms)
        return True
        
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
    
    net_arch = dict(pi=[48, 48], qf=[48, 48])
        
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_starts = 50, # must be at least 2
        buffer_size = 2048,
        learning_rate=3e-4,
        batch_size=64,     # Size of SGD minibatches
        gamma=0.99,        # Discount factor
        verbose=1,
        tensorboard_log = log_dir,
        policy_kwargs = dict(
            net_arch=net_arch,
        ),
    )
    
    # print the nb of parameters
    nb_elements = sum(p.numel() for p in model.policy.parameters())
    print(f"Number of parameters: {nb_elements}")
    
    model.learn(total_timesteps=100_000, progress_bar=True, log_interval=1, callback=InferenceTimeCallback())

if __name__ == "__main__":
    main()