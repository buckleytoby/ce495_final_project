# CE 495 -- Scientific Machine Learning
# Final Project

## Description
Different machine learning algorithms were used to train a robotic autonomous control policy. The policies were compared with respect to their model size, inference time, and peak performance.

## Algorithms
### PPO
A baseline algorithm was chosen to compare performance against. PPO was used because of its high performance and quick convergence time.

PPO is an online algorithm that works by increasing the probability of "good" actions being taken, and decreases the probability of "bad" actions being taken. Good and bad are computed relative to an average which produces an "Advantage".

PPO loss:
$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$


### Actorless Actor Critic
[paper ref]
In standard Soft Actor-Critic, the Critic is trained to predict q-values using the Bellman Equation. The Actor is trained to maximize the expected q-value of its output. This happens by taking the Actor's output, feeding it through the Critic, then using back-propagation to calculate a gradient of the action, and then using that action gradient to update the Actor's weights. 

The SAC Bellman Equation, with an exploration term ($\log \pi(a'|s')$)
$$
Q(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P, a' \sim \pi} \left[ Q(s', a') - \alpha \log \pi(a'|s') \right]
$$

In Actorless Actor-Critic, the Actor is removed during training. Now, during inference, an numerical optimization problem is solved. The gradient for the solver is obtained via back-propagation through the Critic, and the gradient feeds into some optimization algorithm to update the action.

In this work, I used Torch's L-BFGS optimizer to update a randomly initialized action during inference, utilizing the Critic's back-propagated gradient. L-BFGS is a quasi-second order method which uses several iterations to estimate the Hessian of the action, instead of computing it directly. This saves time compared to a proper second order method, like Newton's Method.

For simplicity, I use scheduled epsilon-greedy for exploration, and set the exploration constant, $\alpha$, to zero.

### Neural ODE
A Neural ODE is a neural net which is trained to approximate an ordinary differential equation (or system thereof). The input is the dependent variable, and t, and the output is the gradient of the dependent variable. In our case, I cast the gradient of the Critic from Actorless Actor-Critic as an ODE and train a neural net to predict the gradient of the Critic. This neural ODE is then used during inference to predict action gradients, and once again Torch's L-BFGS solves the optimization problem using those action gradients.

My Neural ODE loss function:
$$
L = \text{MSE} \left( \pi(s, a), \frac{\partial Q(s, a)}{\partial a} \right)
$$

Where $\pi(s,a)$ is the Neural ODE Actor.

The advantage of a Neural ODE over doing backprop on the Critic is that inference should be over 2x faster, since only forward inference must be computed. Backward passes tend to take more than 100% of the forward pass time, so removing it should save substantial inference time.

Similar to SAC, the Neural ODE Actor and the Critic were trained simultaneously.

## Training Setup
The network parameters were chosen so that all policies were roughly the same size, for a fair comparison. The learning rate was set to $3e-4$ for all algorithms.

## Relevance to CE495: Scientific Machine Learning
Two topics covered in CE495 were explored in this project. The first was numerical optimization techniques, in my use of L-BFGS, and the second was the use of Neural ODE's to approximate an ODE.

My hypothesis for this project was that there would be a tradeoff between inference time, and performance. I predicted that if a network takes more time to "think" about its action, then it will produce better actions.

The results, which are reviewed here: [youtube video link]

show that with this project's setup, [talk about SAC vs Neural SAC vs Actorless SAC]

Furthermore, there was not a significant speedup in using a Neural ODE during inference, and the performance dropped significantly with the Neural ODE. This is likely because the Neural ODE Actor wasn't trained long enough on the Critic's gradients to be able to accurately approximate it.

# Run this code
## Requirements
Ubuntu / Linux  
Conda / MiniConda

## Prerequisites
You should already have conda or miniconda installed. Create a new conda env by running the following command from this directory:
```
conda env create -f environment.yml
```

## How To Run
open a terminal
activate the venv:
```
conda activate ce495
```
navigate to the src folder
```
cd src/low_dof_rotate
```

Run any of these commands:
```
python low_dof_rotate_ppo.py
```

```
python low_dof_rotate_sac.py
```

```
python low_dof_rotate_rl_simple_actorless_critic_lbfgs.py
```

```
python low_dof_rotate_rl_fixed_neural_ode.py
```

Open a web browser and go to
```
localhost:7000
```
to watch the drake simulation


If you want to view tensorboard, open a new terminal.   

activate the env in this terminal
```
conda activate ce495
```
and navigate to the puck logs
```
cd src/low_dof_rotate/puck_logs/
```
run
```
tensorboard --logdir .
```

Open a web browser and go to
```
localhost:6006
```
to view tensorboard

## Parallel Envs
There is an option to run many parallel environments to expedite training.

Navigate to the file `low_dof_gym_env.py`. In line 233, change `False` to `True` to enable parallel envs in sub-processes. Change the `n_envs` parameter to a value of your liking (no more than the number of cores in your CPU, though).

Now re-run any of the learning scripts from above.