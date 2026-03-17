

"""
For this project I'm testing several different torch-based methods to do inference with a critic. 

Different methods to test:
baselines:
torch LBFGS
torch hessian
torch adam

compiled policies?
implicit methods?

see this repo:
https://github.com/rtqichen/torchdiffeq
"""


import torch as th
from torch import nn

import diffusion_policy.globals as globals
from diffusion_policy.common.replay_buffer import (
    ReplayBuffer,
)

# params
NB_INPUTS = 10
NB_HIDDEN_FEATURES = 128
INFERENCE_BATCH_SIZE = 64



class MakeDataset:
    def __init__(self) -> None:
        pass


class Inference:
    def __init__(self) -> None:
        pass

class LBFGS(Inference):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def infer(self):
        action = th.normal(0, 1, size=(INFERENCE_BATCH_SIZE, NB_INPUTS,), requires_grad=True)

        def closure():
            qval = self.critic(action)

            # sum the qvals to get a single scalar loss which keeps all samples independent
            loss = -qval.sum()

            return loss
        
        # make the lbfgs optimizer
        optimizer = th.optim.LBFGS([action], line_search_fn='strong_wolfe')

        # run it
        optimizer.step(closure)

class Hessian(Inference):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

class Adam(Inference):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic


class FinalProject:
    def __init__(self) -> None:

        # generate dataset
        self.make_dataset = MakeDataset()

        # create the critic
        self.critic = nn.Sequential(
            nn.Linear(NB_INPUTS, NB_HIDDEN_FEATURES),
            nn.Mish(),
            nn.Linear(NB_HIDDEN_FEATURES, NB_HIDDEN_FEATURES),
            nn.Mish(),
            nn.Linear(NB_HIDDEN_FEATURES, 1)
        )

        # train the critic

        # run the baselines
        lbfgs = LBFGS(self.critic)
        hessian = Hessian(self.critic)
        adam = Adam(self.critic)




if __name__ == "__main__":
    final_project = FinalProject()