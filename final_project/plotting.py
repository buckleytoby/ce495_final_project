

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

# sac.csv
sac = pd.read_csv(os.path.join(data_path, "sac.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
plt.plot(sac["Step"], sac["Value"], label="SAC", color="cyan")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("SAC Reward over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "sac_reward_plot.png"))

# trpo.csv
trpo = pd.read_csv(os.path.join(data_path, "trpo.csv"), sep="\t")

# plot
plt.figure(figsize=(10, 6))
plt.plot(trpo["Step"], trpo["Value"], label="TRPO", color="magenta")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("TRPO Reward over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "trpo_reward_plot.png"))

# sgd
sgd = pd.read_csv(os.path.join(data_path, "sgd.csv"), sep="\t")


# gradient ppo
gradient_ppo = pd.read_csv(os.path.join(data_path, "gradient_ppo.csv"), sep="\t")


nb = 100_000
# nb = 20000

# plot all rewards together for comparison, only plot <100k
good_ppo = good_ppo[good_ppo["Step"] < nb]
decent_lbfgs = decent_lbfgs[decent_lbfgs["Step"] < nb]
neural_ode = neural_ode[neural_ode["Step"] < nb]
sac = sac[sac["Step"] < nb]
trpo = trpo[trpo["Step"] < nb]
sgd = sgd[sgd["Step"] < nb]
gradient_ppo = gradient_ppo[gradient_ppo["Step"] < nb]


plt.figure(figsize=(10, 6))
plt.plot(good_ppo["Step"], good_ppo["Value"], label="PPO", color="blue")
plt.plot(gradient_ppo["Step"], gradient_ppo["Value"], label="Gradient PPO LBFGS", color="purple")
plt.plot(trpo["Step"], trpo["Value"], label="TRPO", color="magenta")
plt.plot(sac["Step"], sac["Value"], label="SAC", color="cyan")
plt.plot(decent_lbfgs["Step"], decent_lbfgs["Value"], label="SAC Critic LBFGS", color="orange")
plt.plot(sgd["Step"], sgd["Value"], label="SAC Critic SGD", color="red")
plt.plot(neural_ode["Step"], neural_ode["Value"], label="SAC Neural ODE LBFGS", color="green")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Reward Comparison over Time")
plt.legend()
plt.grid()

# save it
plt.savefig(os.path.join(plots_path, "reward_comparison_plot.png"))