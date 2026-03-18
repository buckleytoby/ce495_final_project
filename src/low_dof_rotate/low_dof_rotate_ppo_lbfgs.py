

"""
in this version, we use PPO to output delta-actions (aka action gradients), we then use an online numerical optimization solver to produce actions.
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
    
    
class GradientPPOAlgorithm(PPO):
    """
    PPO that outputs action gradients instead of actions.
    
    The PPO actor-critic operate in k-space, while the t-critic operates in t-space.
    
    
    for code re-use, let self.policy be the k-space policy
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        
    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict_lbfgs(observation, deterministic)
        
    def _predict_lbfgs(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor: 
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
            
            def get_action_grad(observation):
                return self.policy.get_distribution(observation).get_actions(deterministic=deterministic)
            
            def closure():
                optimizer.zero_grad()
                
                # get the gradient, no grad to save time
                with th.no_grad():
                    action_grad = get_action_grad(observation)
                
                # define a grad loss as action_tensor @ action_grad. Such that partial (grad_loss) / partial (action_tensor) = action_grad. Do it this way so that LBFGS can properly track the gradients via the loss
                grad_loss = -th.sum(action_tensor * action_grad)
                
                # small l2 reg penalty to avoid shooting off to infinity
                l2_reg = 1e-6 * th.sum(action_tensor**2)
                
                loss = grad_loss + l2_reg
                loss.backward()
                    
                return loss.item()
            
            # setup lbfgs
            optimizer = th.optim.LBFGS([action_tensor], max_iter=5, line_search_fn='strong_wolfe')
            
            # run it
            optimizer.step(closure)
            
            # we're done
            best_action = action_tensor.detach()
        return best_action
    
    def train(self):
        self.train_kspace()
        self.train_tspace(1)
    
    def train_tspace(self, gradient_steps: int) -> None:
        policy = self.t_sac_policy
        critic = policy.critic
        critic_target = policy.critic_target
        batch_size = policy.batch_size
        ent_coef_optimizer = policy.ent_coef_optimizer
        ent_coef = policy.ent_coef
        gamma = policy.gamma
        target_update_interval = policy.target_update_interval
        tau = policy.tau
        batch_norm_stats = get_parameters_by_name(policy.critic, "running_var")
        batch_norm_stats_target = get_parameters_by_name(policy.critic_target, "running_var")
        
        
        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [critic.optimizer]
        if ent_coef_optimizer is not None:
            optimizers += [ent_coef_optimizer]

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
                next_q_values = th.cat(critic_target(replay_data.next_observations, replay_data.next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                
                # td error
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            critic.optimizer.zero_grad()
            critic_loss.backward()
            critic.optimizer.step()
            
            # Update target networks
            if gradient_step % target_update_interval == 0:
                polyak_update(critic.parameters(), critic_target.parameters(), tau)
                # Copy running stats, see GH issue #996
                polyak_update(batch_norm_stats, batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/critic_loss", np.mean(critic_losses))
            
        
        self.logger.dump(step=self._n_updates)
    
    def train_kspace(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        
        # Optional: clip range for the value function
        clip_range_vf = 0.0
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        
        
        # default values to make pylance happy
        approx_kl_divs = []
        loss = th.tensor(0.0)
            
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    
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