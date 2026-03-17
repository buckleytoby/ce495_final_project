

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os, pathlib

data_path = "/Users/grognak/Documents/GitHub/ce495_final_project/final_project/data/"
plots_path = "/Users/grognak/Documents/GitHub/ce495_final_project/final_project/plots"

# good ppo
good_ppo = pd.read_csv(os.path.join(data_path, "good_ppo.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
plt.plot(good_ppo["Step"], good_ppo["Value"], label="PPO", color="blue")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("PPO Reward over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "ppo_reward_plot.png"))


# now for decent_critic_lbfgs
decent_lbfgs = pd.read_csv(os.path.join(data_path, "decent_critic_lbfgs.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
plt.plot(decent_lbfgs["Step"], decent_lbfgs["Value"], label="LBFGS", color="orange")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Critic Gradient Based LBFGS Reward over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "lbfgs_reward_plot.png"))

# neural_ode.csv
neural_ode = pd.read_csv(os.path.join(data_path, "neural_ode.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
plt.plot(neural_ode["Step"], neural_ode["Value"], label="Neural ODE", color="green")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Neural ODE Critic Reward over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "neural_ode_reward_plot.png"))

# neural ode critic loss
neural_ode_loss = pd.read_csv(os.path.join(data_path, "neural_ode_critic_loss.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
plt.plot(neural_ode_loss["Step"], neural_ode_loss["Value"], label="Neural ODE Critic Loss", color="red")
plt.xlabel("Step")
plt.ylabel("Log Loss")
plt.title("Neural ODE Critic Loss over Time")
plt.yscale('log')
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "neural_ode_critic_loss_plot.png"))

# neural actor loss
neural_actor_loss = pd.read_csv(os.path.join(data_path, "neural_ode_actor_loss.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
# plot log loss
plt.plot(neural_actor_loss["Step"], neural_actor_loss["Value"], label="Neural ODE Actor Loss (log scale)", color="purple")
plt.xlabel("Step")
plt.ylabel("Log Loss")
plt.title("Neural ODE Actor Loss over Time")
plt.yscale('log')
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "neural_ode_actor_loss_plot.png"))


# plot all rewards together for comparison
plt.figure(figsize=(10, 6))
plt.plot(good_ppo["Step"], good_ppo["Value"], label="PPO", color="blue")
plt.plot(decent_lbfgs["Step"], decent_lbfgs["Value"], label="Critic LBFGS", color="orange")
plt.plot(neural_ode["Step"], neural_ode["Value"], label="Neural ODE LBFGS", color="green")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Comparison over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "reward_comparison_plot.png"))