




import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ContinuousCritic, BasePolicy


from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union

from stable_baselines3.common.envs import SimpleMultiObsEnv # Example dict env

from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
#
import torch as th
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Type, Union


from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.sac.policies import MultiInputPolicy as SACMultiInputPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, DictReplayBufferSamples
#
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


from low_dof_rotate_sim import LowDOFRotateSim


# Assuming your environment setup is in a file named low_dof_env_factory.py
# or defined earlier in your script
from low_dof_gym_env import LowDOFRotateGymEnv

import gymnasium as gym

from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer



from pydrake.multibody.plant import MultibodyPlant
from pydrake.autodiffutils import AutoDiffXd, InitializeAutoDiff, ExtractGradient, ExtractValue
import numpy as np
import torch as th


from simple_scheduling import (
    SimpleLinearScheduler,
)


##################################################

NUM_ACTIONS = 3 # ff
    
def low_dof_action_step(observation, action):
    # take a small step in the direction of the action for the pose
    if isinstance(observation, dict):
        joint_state = observation['joint_state']
    else:
        joint_state = observation
    
    action = joint_state + 1.0 * action
    
    return action



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

# need to make it a policy
class ActorlessCriticPolicy(SACPolicy):
    """
    forward and _predict must produce actions
    """
    # alias
    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def inference(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict_sgd(obs, deterministic=deterministic)
    
    # override to not use the actor
    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict_sgd(observation, deterministic=deterministic)

    def _predict_sgd(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor: 
        """
        predict using lbfgs
        """
        with th.enable_grad():
            assert(isinstance(observation, th.Tensor))
            nb_envs = observation.shape[0]
            Da = self.action_space.shape[0] # type:ignore
            
            # make the action tensor to be optimized
            action_tensor = th.randn((nb_envs, Da), requires_grad=True, device=self.device)
            initial_action = action_tensor.clone().detach()
            
            # simple SGD optimizer to optimize the action tensor
            optimizer = th.optim.SGD([action_tensor], lr=0.1)
            
            for _ in range(10): # number of optimization steps
                optimizer.zero_grad()
                
                # get the q value from the critic
                q_values = self.critic(observation, action_tensor)
                q_value = q_values[0] # type:ignore
                
                # small l2 reg penalty to avoid shooting off to infinity
                l2_reg = 1e-8 * th.sum(action_tensor**2)
                
                loss = -q_value.mean() + l2_reg
                
                loss.backward()
                optimizer.step()
        
        # we're done
        best_action = action_tensor.detach()
        return best_action
        
##
#
#
# actorless critic algorithm (off policy alg)
class ActorlessCriticAlgorithm(SAC):
    """
    No actor, just critic.
    
    Also no entropy loss, I want to do a custom exploration strategy
    """
    def __init__(self, 
                 exploration_rate_random = 0.66,
                #  exploration_rate_metric = 0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.exploration_rate_random = exploration_rate_random
        # self.exploration_rate_metric = exploration_rate_metric
        
        
        self.exploration_rate_random_schedule = SimpleLinearScheduler(0, self.exploration_rate_random, 100_000, 0.0)
        
    # @override
    def _create_aliases(self) -> None:
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        
    # @override
    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "critic.optimizer"]
        saved_pytorch_variables = []

        return state_dicts, saved_pytorch_variables

        
    def optimal_policy_action_gpu_impl(self, observation_gpu):
        self.policy: ActorlessCriticPolicy #type:ignore
        action = self.policy.inference(observation_gpu)
        return action
        
    def optimal_policy_action(self, observation: dict | np.ndarray):
        """
        input and output are CPU
        """
        # move each observation to torch
        if isinstance(observation, dict):
            obs = {key: th.as_tensor(observation[key]).to(self.device).float() for key in observation}
        else:
            obs = th.as_tensor(observation).to(self.device).float()
        
        final_action_tensor = self.optimal_policy_action_gpu_impl(obs)
        
        assert(final_action_tensor.shape[1] == NUM_ACTIONS) # action dim
        
        # to cpu, to numpy, squeeze
        final_action_np = final_action_tensor.cpu().numpy()
        
        # cpu
        return final_action_np
    
    def optimal_policy_action_gpu(self, observation):
        """
        same as optimal_policy_action but with gpu tensors. observation is already on device
        
        input and output on DEVICE
        """
        obs_dict = observation
        
        final_action = self.optimal_policy_action_gpu_impl(obs_dict)
        
        assert(final_action.shape[0] == 64)
        assert(final_action.shape[1] == NUM_ACTIONS) # action dim
        
        return final_action
    
    def get_dist_loss(self, position, target_position):
        return F.mse_loss(position, target_position)
    

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """
        SAC train, overriden, no entropy
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        critic_losses = []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            self.replay_buffer: ReplayBufferWithNextAction #type:ignore
            replay_data: DictReplayBufferSamples
            
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Select action according to policy
                # next_actions = self.optimal_policy_action_gpu(replay_data.next_observations)
                # next_actions = replay_data.next_actions
                
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, replay_data.next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                
                # td error
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
            
        
        self.logger.dump(step=self._n_updates)
    

    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # assert(isinstance(observation, dict))
        
        if isinstance(observation, np.ndarray):
            nb_envs = observation.shape[0]
        else:
            nb_envs = observation[next(iter(observation.keys()))].shape[0]
        
        if deterministic:
            action, state = self.policy.predict(observation, state, episode_start, deterministic)
            return action, state
        
        n = np.random.rand()
        
        
        # random action
        if n < self.exploration_rate_random_schedule.get_value(self._n_updates):
            if self.policy.is_vectorized_observation(observation):
                if isinstance(observation, dict):
                    n_batch = observation[next(iter(observation.keys()))].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
                
                
            action = low_dof_action_step(observation, action)
                
                
        # optimal action w.r.t. the policy
        else:
            action, state = self.optimal_policy_action(observation), None
            
            
        # # optimal action w.r.t. some metric
        # else:
        #     action, state = self.optimal_metric_action_low_dof(observation), None
            
        #     # add env dim, required for "is_vectorized_observation"
        #     action = np.expand_dims(action, axis=0)
            
        assert(action.shape[0] == nb_envs)
        assert(action.shape[1] == NUM_ACTIONS) # action dim
        
        return action, state
    
    
from stable_baselines3.common.callbacks import BaseCallback
import time

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
    env = LowDOFRotateGymEnv()
    
    # 2. Wrap the env for monitoring and SB3 compatibility
    # Monitor logs episode rewards and lengths
    
    # wrap the env in the action flattener
    if False:
        env = FlattenActionWrapper(env)

    # 3. Initialize the Agent
    # We use MlpPolicy because the state is likely a small vector
    # (joint position, velocity, and maybe contact wrenches).
    
    log_dir = "./puck_logs/"
    
    net_arch = dict(pi=[0], qf=[48, 48])
    
    model = ActorlessCriticAlgorithm(
        policy = ActorlessCriticPolicy,
        env=env,
        learning_starts = 50, # must be at least 2
        buffer_size = 2048,
        learning_rate=3e-4,
        batch_size=64,     # Size of SGD minibatches
        gamma=0.99,        # Discount factor
        verbose=1,
        replay_buffer_class = ReplayBufferWithNextAction,
        tensorboard_log = log_dir,
        policy_kwargs = dict(
            net_arch=net_arch,
        ),
        
    )
    
    PROFILING = True
    if PROFILING:
        cb = InferenceTimeCallback()
    else:
        cb = None
    
    
    # print the nb of parameters
    nb_elements = sum(p.numel() for p in model.policy.parameters())
    print(f"Number of parameters: {nb_elements}")

    # 4. Train the agent
    total_timesteps = 200_000
    print(f"Starting training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, log_interval=24, callback=cb)


if __name__ == "__main__":
    main()