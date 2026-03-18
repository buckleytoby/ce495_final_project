

"""
in this version, we use PPO to output delta-actions (aka action gradients), we then use an online numerical optimization solver to produce actions.
"""






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

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm

from stable_baselines3.common.envs import 

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
    
class GradientPPOAlgorithm(PPO):
    """
    PPO that outputs action gradients instead of actions.
    
    The PPO actor-critic operate in k-space, while the t-critic operates in t-space.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # must also create a t-space critic.
        self.t_critic = self.policy.make_critic()
        
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Get the action gradients from the policy
        action_gradients, _ = self.policy.forward(observation)
        
        # Here you would implement your optimization logic to convert the action gradients
        # into actual actions. This is a placeholder and should be replaced with your solver.
        optimized_actions = self.optimize_actions(action_gradients)
        
        return optimized_actions
    
    def optimize_actions(self, action_gradients: th.Tensor) -> th.Tensor:
        # Placeholder for your optimization logic
        # For example, you could use a simple gradient ascent step or a more complex solver.
        optimized_actions = action_gradients  # Replace with actual optimization
        return optimized_actions
        
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
    
    # print the nb of parameters
    nb_elements = sum(p.numel() for p in model.policy.parameters())
    print(f"Number of parameters: {nb_elements}")
    
    model.learn(total_timesteps=100_000, progress_bar=True, log_interval=1, callback=InferenceTimeCallback())

if __name__ == "__main__":
    main()