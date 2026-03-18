import numpy as np
import pydrake.all
from pydrake.all import LeafSystem, AbstractValue
from pydrake.math import RigidTransform

# needed for Diffusion repo compat
import replay_buffer as replay_buffer
# import ReplayBuffer

import utils

class SimpleStateSaver(LeafSystem):
    """
    super simple functionality
    """
    def __init__(self,
                 data_period = 0.1,
                 file_path = "data/simple_state_saver.zarr",
    ):
        LeafSystem.__init__(self)
        
        self.data_period = data_period
        self.file_path = file_path
        
        # my members
        self.replay_buffer = replay_buffer.ReplayBuffer.create_from_path(self.file_path, mode='a')
        
        # print some replay buffer stats
        print("Replay buffer initialized at path: {}".format(self.file_path))
        print("Nb episodes in buffer: {}".format(self.replay_buffer.n_episodes))
        print("Total nb datapoints in buffer: {}".format(self.replay_buffer.__len__()))
        
        ## only output ports / periodic events, no inputs (that should be handled by the inheriting class)
        # dummy output port to allow triggering of flushing the buffer
        self.DeclareVectorOutputPort("flusher", 1, calc = self.DumpEpisode)
        
        # Register the Periodic Event (0.1 seconds = 100ms)
        self.DeclarePeriodicPublishEvent(
            period_sec = self.data_period,
            offset_sec = 0.0, 
            publish = self.RecordState
        )
        
        # dummy output port for data saving trigger
        self.DeclareVectorOutputPort("record_state_trigger", 1, calc = self.RecordStateTrigger)
        
        self.reset()
        
    def reset(self):
        # TODO: make episode a context variable so parallel envs don't interfere? Not an issue for HITL sim, lol nvm
        self.episode = {}
        
    def assemble_datapt(self, context):
        """
        must be implemented by the child
        """
        raise NotImplementedError
    
    def RecordState(self, context):
        """
        This method is called exactly once every 100ms.
        Will also be called in sim.Initialize()
        
        context - plant.get_body_poses_output_port
        """
        # start of sim time safety check
        # current_sim_time = context.get_time()
        # if current_sim_time < 1.0:
        #     return
        
        # construct the data dict for this data point
        datapt_dict = self.assemble_datapt(context)
        
        # get timestamp
        t = context.get_time()
        
        # round to the nearest 0.1s
        t = round(t / self.data_period) * self.data_period
        
        # convert to string
        t_str = "{:.1f}".format(t)
        
        # no data protection
        if datapt_dict is not None:
            # add to episode. this will overwrite the previous entry for t_str if one exists.
            self.episode[t_str] = datapt_dict
            
    def RecordStateTrigger(self, context, output):
        self.RecordState(context)
    
    def DumpEpisode(self, context, output):
        """
        this resets myself too
        """
        
        # data struct
        data_dict = dict()
        
        # convert self.episode dict to an episode list
        episode_list = list(self.episode.values())
        
        # stack data
        for key in episode_list[0].keys():
            data_dict[key] = np.stack([x[key] for x in episode_list])
            
        # add to replay buffer. Safe to ctrl+C after it completes
        self.replay_buffer.add_episode(data_dict, compressors='disk')
        
        print("Episode data saved. Nb eps: {}, Ep len: {}, Dataset len: {}".format(self.replay_buffer.n_episodes, len(episode_list), self.replay_buffer.__len__()))
        
        # reset ep
        self.reset()
        

class StateSaver(LeafSystem):
    def __init__(self, plant, body_index, data_period = 0.1, file_path="pose_history.zarr"):
        LeafSystem.__init__(self)
        self.plant = plant
        self.body_index = body_index
        self.data_period = data_period
        self.file_path = file_path
        
        # my members
        self.replay_buffer = replay_buffer.ReplayBuffer.create_from_path(self.file_path, mode='a')
        
        # print some replay buffer stats
        print("Replay buffer initialized at path: {}".format(self.file_path))
        print("Nb episodes in buffer: {}".format(self.replay_buffer.n_episodes))
        print("Total nb datapoints in buffer: {}".format(self.replay_buffer.__len__()))
        
        
        self.reset()
        
    def create_ports(self):
        # Input Port: Body Poses
        self.pose_port = self.DeclareAbstractInputPort(
            "body_poses", 
            self.plant.get_body_poses_output_port().Allocate()
        )
        
        # input port: episode end trigger
        self.trigger_input = self.DeclareVectorInputPort("trigger", 1)
        
        # dummy output port to allow flushing of the buffer
        self.DeclareVectorOutputPort("flusher", 1, calc = self.DumpEpisode)
        
        # output port for current data-point
        self.DeclareAbstractOutputPort(
            "current_datapt",
            lambda: AbstractValue.Make(dict()),
            self.calc_current_datapt
        )
        
        # Register the Periodic Event (0.1 seconds = 100ms)
        self.DeclarePeriodicPublishEvent(
            period_sec = self.data_period,
            offset_sec = 0.0, 
            publish = self.RecordState
        )
        
    def reset(self):
        self.episode = []
        self.old_state = None
        
    def get_poses(self, context) -> list[RigidTransform]:
        # Evaluate poses
        poses = self.pose_port.Eval(context)
        
        assert(isinstance(poses, list))
        return poses
    
    def add_object_to_vector(self, idx, pose, vector):
        
        # add object id
        vector = np.concatenate([vector, [idx]])
        
        # add state
        vector = utils.add_pose_to_vector(pose, vector)
        
        return vector
        
    def save_state(self, context):
        raise NotImplementedError
        
    def get_manipulanda_pose_from_state(self, state):
        # skip the body_index, get x, y, z, qw, qx, qy, qz
        pose = state[1: 1 + 7]
        return pose
        
    def check_idle(self, action, state):        
        pose = self.get_manipulanda_pose_from_state(state)
    
        idle = np.all(action == pose)
        
        return idle
        
    def assemble_datapt(self, context):
        return None
        

    def RecordState(self, context):
        """
        This method is called exactly once every 100ms.
        
        context - plant.get_body_poses_output_port
        """
        
        # construct the data dict for this data point
        datapt_dict = self.assemble_datapt(context)
        
        # no data protection
        if datapt_dict is not None:
            # add to episode
            self.episode.append(datapt_dict)
        
    def DumpEpisode(self, context, output):
        # empty episode protection
        if len(self.episode) < 10:
            print("Too few datapoints in ep. Skipping.")
            return
        
        # data struct
        data_dict = dict()
        
        # stack data
        for key in self.episode[0].keys():
            data_dict[key] = np.stack([x[key] for x in self.episode])
            
        # add to replay buffer. Safe to ctrl+C after it completes
        self.replay_buffer.add_episode(data_dict, compressors='disk')
        
        print("Episode data saved. Nb eps: {}, Ep len: {}, Dataset len: {}".format(self.replay_buffer.n_episodes, len(self.episode), self.replay_buffer.__len__()))
        
        # reset ep
        self.reset()
    
    def wire_upstream(self, builder):
        # plant state --> state saver
        builder.Connect(
            self.plant.get_body_poses_output_port(),
            self.GetInputPort("body_poses")
        )
        
    def calc_current_datapt(self, context, output):
        # construct the data dict for this data point
        datapt_dict = self.assemble_datapt(context)
        
        if datapt_dict is None:
            # empty dict
            output.set_value(dict())
        else:
            output.set_value(datapt_dict)
        
        
        
class ObjectPilotStateSaver(StateSaver):
        
    def save_state(self, context):
        """
        save the current state
        """
        poses = self.get_poses(context)
        
        # state vector
        vector = np.empty(0)
        
        # start with the manipulanda
        vector = self.add_object_to_vector(int(self.body_index), poses[self.body_index], vector)
        
        # iterate over all object poses
        for idx, pose in enumerate(poses):
            vector = self.add_object_to_vector(idx, pose, vector)
        
        self.old_state = vector
        
    def assemble_datapt(self, context): #type:ignore
        """
        NOTE: TODO: this will record the first data-point, when the objects are teleported to their randomized start location. Consider adding a way to denote an episode beginning (after randomization is complete).
        """
        # first step
        if self.old_state is None:
            # still gotta save the current state
            self.save_state(context)
            return None
        
        # keys must be consistent with the yaml
        # state will be the previous state
        state = self.old_state.copy()
        
        # action is from the current state, so first save the current state
        self.save_state(context)
        
        # extract action
        action = self.get_manipulanda_pose_from_state(self.old_state).copy()
        
        # idle protection?
        if True:
            if self.check_idle(action, state):
                return None
        
        # assemble final dict
        datapt = {
            'state': state,
            'action': action,
            'not_done': True,
            'reward': 0.0,
            'task_id': 20, # from dexnex_projects/.../tasks.md
            'qval': 0.0,
        }
        return datapt