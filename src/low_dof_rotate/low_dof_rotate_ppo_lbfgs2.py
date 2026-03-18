

"""
in this version, we use PPO to output delta-actions (aka action gradients), we then use an online numerical optimization solver to produce actions.

I think this may still suffer from the issue of exploiting the critic, because ultimately I'm using a t-space critic to provide rewards. This could be improved by using GAE to provide the reward (basically don't update the k-space PPO until an entire t-space episode completes)

"""






import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
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


import time

import torch as th


from stable_baselines3.common.type_aliases import PyTorchObs

from stable_baselines3.common.utils import FloatSchedule, explained_variance

from gymnasium import spaces

from stable_baselines3.sac.policies import SACPolicy

from stable_baselines3.common.utils import get_parameters_by_name, polyak_update

from torch.nn import functional as F

from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, DictReplayBufferSamples

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer, RolloutBuffer

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from typing import Any, Optional, TypeVar, Union
SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.env_checker import check_env

class ReplayBufferSamplesWithNextAction(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    # For n-step replay buffer
    discounts: Optional[th.Tensor] = None

# class ReplayBufferWithNextAction(DictReplayBuffer):
class ReplayBufferWithNextAction(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add extra storage for next actions
        self.next_actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        # We need the action that was just taken to be stored 
        # as the 'next_action' for the PREVIOUS transition.
        # This requires careful indexing.
        
        # calc previous pos before .add because that increments self.pos
        previous_pos = (self.pos - 1) % self.buffer_size
        current_pos = self.pos % self.buffer_size
        
        # call the super, this increments self.pos so DON"T USE IT
        super().add(obs, next_obs, action, reward, done, infos)
        
        # get the stored action
        action = self.actions[current_pos]
        
        # Store the current action as the next_action for the previous step
        self.next_actions[previous_pos] = np.array(action).copy()
        
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None): #type:ignore
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        # just note, need at least 2 transitions for a single training sample.
        
        upper_bound = self.buffer_size if self.full else self.pos
        
        # max is upper_bound - 2 because we must return next actions, which we'll only have for upper_bound-2. It's 2 instead of 1 because .add increments self.pos
        upper_bound -= 2
        
        # ah, upper bound needs to be one more than the biggest index we want to sample. it's [low, high)
        upper_bound += 1
        
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env=None): #type:ignore
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            self.next_actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        
        return ReplayBufferSamplesWithNextAction(*tuple(map(self.to_torch, data))) #type:ignore

class FlattenActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Manually define a flat space (e.g., Box) that covers all dict values
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape=(7+22,)) # (7+22+5+3*5+7+7,) for obs

    def action(self, action):
        # Unflatten the vector back into the dictionary format your env expects
        return {"pose_action": action[0:7], "joint_action": action[7:29]}


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
    

    
class TCriticAlgorithm(SAC):
    """
    the actor is actually another algorithm
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        log_dir = "./puck_logs/"
        
        # my parameters
        self.max_nb_iterations = 10
        self.nb_kspace_warmup_steps = 100
        self.iteration = 0
        
        self.nb_envs = self.env.num_envs if self.env is not None else 1
        self.t_observation = th.zeros((self.nb_envs, self.observation_space.sample().shape[0]))
                
            
        # instantiate the k-space env
        k_space_env = KSpaceEnv(self)
        self.k_env_needs_reset = True
        
        # make a child PPO algorithm to be the actor
        assert(self.env is not None)
        self.ppo = KSpacePPOAlgorithm(
            policy="MlpPolicy",
            env=k_space_env,
            learning_rate=3e-4,
            n_steps=self.max_nb_iterations,      # for k-space PPO, this MUST be equal to max nb iterations
            batch_size=self.max_nb_iterations,     # Size of SGD minibatches
            n_epochs=10,       # Optimization epochs per update
            gamma=0.99,        # Discount factor
            verbose=1,
            tensorboard_log=log_dir,
        )
        
    def learn(self, *args, **kwargs):
        # setup ppo
        self.ppo.learn_setup(total_timesteps = 100_000 * self.max_nb_iterations, callback = None, log_interval=self.max_nb_iterations, progress_bar=True)
        
        # learn
        learned = super().learn(*args, **kwargs)
        
        # post learn
        self.ppo.learn_post(total_timesteps = 100_000 * self.max_nb_iterations, log_interval=self.max_nb_iterations, progress_bar=True)
        
        return learned
    
    # override
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        our "prediction" is actually a rollout for the k-space algorithm.
        """
        self.iteration += 1
        if self.iteration < self.nb_kspace_warmup_steps:
            # just return a random action for a while to let the critic learn something reasonable before we start optimizing actions
            unscaled_action = np.array([self.action_space.sample() for _ in range(self.nb_envs)])
            return unscaled_action, None # 2nd arg not used
        
        else:
            ppo = self.ppo
            assert(isinstance(ppo.get_env().envs[0].unwrapped, KSpaceEnv)) #type:ignore
            
            # save our observation as the t-observation for the k-space env
            self.t_observation = th.from_numpy(observation).to(self.device)
            
            # tell the env that it needs to reset
            self.k_env_needs_reset = True
            
            ppo.learn_one_step()
            
            # at this point, the k-space env should have the optimized t-action stored as self.t_action
            optimized_t_action = ppo.get_env().get_attr("t_action") #type:ignore
            optimized_t_action = th.cat(optimized_t_action, dim=0)
            optimized_t_action = th.tensor(optimized_t_action).to(self.device)
            
            return optimized_t_action.cpu().numpy(), None # 2nd arg not used
        
# make a K-Space env so that PPO doesn't mess up the real environment
class KSpaceEnv(gym.Env):
    def __init__(self, t_alg: TCriticAlgorithm):
        super().__init__()
        self.t_alg = t_alg
        
        
        # my members
        assert(self.t_alg.env is not None)
        self.observation_space = self.t_alg.env.observation_space
        self.action_space = self.t_alg.env.action_space
        
        assert(self.observation_space is not None)
        assert(self.action_space is not None)
        
        assert(self.observation_space.shape is not None)
        assert(self.action_space.shape is not None)
        
        # combine the nb actions and nb observations
        nb_actions_and_obs = self.observation_space.shape[0] + self.action_space.shape[0]
        
        # update the obs space. Recall that in this env, the observation is the t-obs and the t-action, and the action is the k-action.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(nb_actions_and_obs,), dtype=np.float32)
        
        self.max_nb_iterations = self.t_alg.max_nb_iterations
        self.t_observation = th.zeros(self.observation_space.sample().shape)
        self.t_action = th.zeros(self.action_space.sample().shape) 
        
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        """
        save the t-observation and the t-action
        """
        nb_envs = self.t_alg.t_observation.shape[0]
        
        self.t_observation = self.t_alg.t_observation.to(self.t_alg.device)
        self.nb_iterations = 0
        
        # nan protection
        self.t_observation = th.nan_to_num(self.t_observation, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # randomize the taction
        samples = np.array([self.action_space.sample() for _ in range(nb_envs)])
        self.t_action = th.from_numpy(samples).to(self.t_alg.device)
        self.t_action.requires_grad = True
        
        # make a new lbfgs. Set max_iter to 1 so we can do one step at a time
        self.lbfgs = th.optim.LBFGS([self.t_action], max_iter=1, line_search_fn="strong_wolfe")
        
        obs = self.get_observation()
        return obs, dict()
    
    def get_observation(self):
        """
        cat the obs and the t-action
        """
        obs = self.t_observation
        t_action = self.t_action
        
        # cat them together
        observation = th.cat([obs, t_action], dim=1)
        
        return observation.detach().cpu().numpy()
        
    
    def render(self):
        pass
        
    def update_t_action(self, action_gradient):
        action_gradient = th.Tensor(action_gradient).to(self.t_alg.device)
        # this is the closure that LBFGS will optimize over
        def closure():
            self.lbfgs.zero_grad()
                                
            # define a grad loss as action_tensor @ action_grad. Such that partial (grad_loss) / partial (action_tensor) = action_grad. Do it this way so that LBFGS can properly track the gradients via the loss
            grad_loss = -th.sum(self.t_action * action_gradient)
            
            # small l2 reg penalty to avoid shooting off to infinity
            l2_reg = 1e-6 * th.sum(self.t_action**2)
            
            loss = grad_loss + l2_reg
            loss.backward()
            
            # Clip gradients to prevent huge updates
            th.nn.utils.clip_grad_norm_([action_gradient], max_norm=10.0)
                
            return loss.item()
        
        self.lbfgs.step(closure)
        
        # nan protection
        with th.no_grad():
            # Keep it within a reasonable range for the neural network
            self.t_action.clamp_(-10.0, 10.0) 
            # Then catch any rogue NaNs just in case
            self.t_action.nan_to_num_(nan=0.0)
            
    def step(
        self, action
    ):
        """
        action = t-space action gradient use it with some optimizer to update the action (either sgd or lbfgs, for example)
        """
        # first check if a reset was triggered
        if self.t_alg.k_env_needs_reset:
            self.t_alg.k_env_needs_reset = False
            return self.get_observation(), 0, False, True, {}
        
        # get the initial q-val
        q_val_before = self.t_alg.critic.q1_forward(self.t_observation, self.t_action)
        
        # update taction
        self.update_t_action(action)
        
        # get the final q-val
        q_val_after = self.t_alg.critic.q1_forward(self.t_observation, self.t_action)
        
        # reward is the improvement in q-val
        reward = q_val_after - q_val_before
        
        # increment iters
        self.nb_iterations += 1
        
        # check if done
        terminated = False
        if self.nb_iterations >= self.max_nb_iterations:
            terminated = True
            
        # set info
        info = {}
        
        # get the obs
        observation = self.get_observation()
        
        # we're done
        return observation, reward, terminated, False, info
        
        
SelfKSpacePPOAlgorithm = TypeVar("SelfKSpacePPOAlgorithm", bound="KSpacePPOAlgorithm")
class KSpacePPOAlgorithm(PPO): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # my members
        self.iteration = 0
        self.k_space_ppo_warmup = 100
        # self.callback = None
        
    def learn_setup(
        self: SelfKSpacePPOAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "KSpacePPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfKSpacePPOAlgorithm:
        
        self.iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())
        
        self.total_timesteps = total_timesteps
        self.callback = callback

        assert self.env is not None
        
        return self
    
    def learn_post(
        self: SelfKSpacePPOAlgorithm,
        total_timesteps: int,
        log_interval: int = 1,
        tb_log_name: str = "KSpacePPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfKSpacePPOAlgorithm:
        
        self.callback.on_training_end()
        
        return self

        
    def learn_one_step(
        self: SelfKSpacePPOAlgorithm,
        log_interval: int = 1,
        tb_log_name: str = "KSpacePPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfKSpacePPOAlgorithm:
        
        
        assert(self.env is not None)
        
        continue_training = self.collect_rollouts(self.env, self.callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

        self.iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, self.total_timesteps)

        # Display training infos
        if log_interval is not None and self.iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            self.dump_logs(self.iteration)

        self.train()
        return self
        
    
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
        
    net_arch = dict(pi=[0], qf=[48, 48])
        
    model = TCriticAlgorithm(
        policy="MlpPolicy",
        env=env,
        learning_starts = 10, # must be at least 2
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
    
    cb = None if True else InferenceTimeCallback()
    model.learn(total_timesteps=100_000, progress_bar=True, log_interval=1, callback=cb, tb_log_name="TCriticAlgorithm")

if __name__ == "__main__":
    main()