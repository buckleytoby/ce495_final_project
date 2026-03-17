# CE 495 -- Scientific Machine Learning
# Final Project

## Description


## Algorithms
### PPO


### Actorless Actor Critic
[paper ref]

## Requirements
Ubuntu / Linux  
Conda / MiniConda

## Prerequisites
You should already have conda or miniconda installed. Create a new conda env by running the following command:
```
conda env create -f environment.yml
```

## How To Run
open a terminal
activate the venv:
```
conda activate ce495
```
cd to the src folder
```
cd src/low_dof_rotate
```

Run any of these commands:
```
python low_dof_rotate_ppo.py
```

```
python low_dof_rotate_rl_simple_actorless_critic_lbfgs.py
```

```
python low_dof_rotate_rl_fixed_neural_ode.py
```

Go to
```
localhost:7000
```
to watch the drake simulation


If you want to view tensorboard, navigate to the puck logs
```
cd puck_logs/
```
run
```
tensorboard --logdir .
```
Go to 
```
localhost:6006
```
to view tensorboard