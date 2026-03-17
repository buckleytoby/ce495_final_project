

from stable_baselines3.common.monitor import Monitor


import os


from .low_dof_rotate_sim import LowDOFRotateSim

# NOT using our custom DrakeGymEnv
from drake_gym.drake_gym import DrakeGymEnv

import gymnasium as gym
import numpy as np

# import action obs
from .low_dof_gym_env_leafs import LowDOFRotateAction, LowDOFRotateObservation

from pydrake.all import EventStatus


from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

class LowDofRL:
    """
    add on the sandbox sim RL components
    """
    def __init__(self,
                 sim_class: LowDOFRotateSim,
                 timeout = 5.0,
                ):
        self.sim_class = sim_class
        self.timeout = timeout
        
        # refs
        self.plant = self.sim_class.plant
        self.builder = self.sim_class.builder
        
        
        
    def make_leafs(self):        
        # actions coming from GYM
        self.action_subscriber = LowDOFRotateAction()
        
        # observations coming from DRAKE -- this is already done in the state saver
        self.observations = LowDOFRotateObservation()
    
        # reward from the sim class
        self.reward = self.sim_class.reward_leaf
        
    def add_leafs(self):
        # actions
        self.builder.AddSystem(self.action_subscriber)
        
        # observation
        self.builder.AddSystem(self.observations)
        
        ### export ports required for GYM
        # actions from GYM to be consumed by DRAKE
        self.builder.ExportInput(self.action_subscriber.GetInputPort("action_input"), "action")
        self.builder.ExportOutput(self.observations.GetOutputPort("observation"), "observation")
        self.builder.ExportOutput(self.reward.GetOutputPort("reward"), "reward")
        
    def create_ports(self):
        pass
        
    def wire_leafs(self):
        # wire inputs to observation leaf
        self.observations.wire_upstream(
            self.builder,
            self.sim_class.state_saver.GetOutputPort("current_datapt"),
        )
        
        # wire inputs to action leaf
        self.action_subscriber.wire_upstream(
            self.builder,
        )
        
        # joint action port --> joint sub extractor
        self.sim_class.joint_sub_extractor_leaf.wire_upstream(
            self.builder,
            self.action_subscriber.GetOutputPort("joint_action_output"),
        )
        
    def set_home(self, simulator, context, seed):
        """
        when doing RL, we call reset through drake-gym instead of through the sim class directly (in LowDOFRotateSim.sim_monitor)
        """
        self.sim_class.reset()
        
    def has_timed_out(self, root_context):
        """
        check if the episode has timed out
        """
        # get simulation time
        context = self.sim_class.simulator.get_context()
        time = context.get_time()
        
        return time >= self.timeout
    
    # def has_left_bounding_box(self, root_context):
    
    #     """
    #     whether the hand has left the bounding box. Not a huge deal because it shouldn't do this much after a little bit of training
    #     """
    #     context = self.sim_class.simulator.get_context()
    #     hand_position = self.sim_class.floating_body_controller_leaf.get_hand_position(context)
        
    #     # bounding box limits
    #     x_limit = 0.5
    #     y_limit = 0.5
    #     z_limit = 0.5
        
    #     if (abs(hand_position[0]) > x_limit or
    #         abs(hand_position[1]) > y_limit or
    #         abs(hand_position[2]) > z_limit):
    #         return True
    #     else:
    #         return False
        
    def rl_sim_monitor(self, root_context):
        """
        specialized sim monitor for RL. This one doesn't reset and doesn't save data to the RB, because those are both handled in either drake-gym or the RL pipeline (diffusion_policy...)
        """
        # normal status
        event_status = EventStatus.Succeeded()
        
        must_reset = self.sim_class.calc_must_reset(root_context)
            
        # then reset
        if must_reset:
            # set termination flag
            event_status = EventStatus.ReachedTermination(self.sim_class.diagram, "Episode ended")
            
        # check for timeout
        if self.has_timed_out(root_context):
            # set termination flag
            event_status = EventStatus.ReachedTermination(self.sim_class.diagram, "Episode timed out")
            
        return event_status
        
def make_the_sim_class():
    
    # 1. Initialize the simulation object
    sim = LowDOFRotateSim(must_connect_ros2 = False)
    rl = LowDofRL(sim)
    
    # setup the plant
    sim.setup_plant()
    
    # make leafs
    sim.make_leafs()
    rl.make_leafs()
    
    ## finalize plant
    sim.plant.Finalize()
    
    # some ports depend on a finalized plant, so must be called after Finalize
    sim.create_ports()
    rl.create_ports()
    
    # meshcat
    sim.setup_meshcat()
    
    # add all leafs
    sim.add_leafs()
    rl.add_leafs()
    
    # wire leafs
    sim.wire_leafs()
    rl.wire_leafs()
    
    # mux 
    sim.setup_force_mux()
    
    # setup sim
    sim.setup_simulator()
    
    # randomizer
    sim.setup_randomizers()
    
    return sim, rl
        
def LowDOFRotateGymEnv():
    
    def make_drake_env():
        # Create Drake simulation
        sim_class, rl = make_the_sim_class()
        
        # sim
        sim = sim_class.simulator
        
        # override the monitor with the RL sim monitor
        sim.set_monitor(rl.rl_sim_monitor)

        # ehh hard code for now
        nb_joints = 3
        
        action_space = gym.spaces.Box(
                    low = -1.0 * np.ones(nb_joints),
                    high = 1.0 * np.ones(nb_joints),
                    dtype = np.float32
                )
            
        observation_space = gym.spaces.Box(
                    low = -np.inf * np.ones(nb_joints),
                    high = np.inf * np.ones(nb_joints),
                    dtype = np.float32
                )
        
        # params
        time_step = 0.1 # same as everywhere else
        
        env = DrakeGymEnv(simulator = sim,
                        time_step = time_step,
                        action_space = action_space,
                        observation_space = observation_space,
                        reward = "reward",
                        action_port_id = "action",
                        observation_port_id = "observation",
                        set_home = rl.set_home, # used to randomize the domain. docs are wrong, it needs (simulator, context, seed) as input #type:ignore
                        )
        

        log_dir = "./puck_logs/"
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env)
        
        return env
        
    if True:
        env = make_vec_env(make_drake_env, n_envs=24, vec_env_cls=SubprocVecEnv)
    else:
        env = make_vec_env(make_drake_env, n_envs=1, vec_env_cls=DummyVecEnv)
    
    # env = DummyVecEnv([make_drake_env])


    return env

def main():
    # test the env
    env = LowDOFRotateGymEnv()
    print("Env created")
    
if __name__ == "__main__":
    main()