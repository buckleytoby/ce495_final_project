"""
drake leaf system which takes a dictionary input port with keys 'joint_action' and 'pose_action' and separates them into two output ports.

basically a dictionary demux
"""

from pydrake.systems.framework import LeafSystem, BasicVector
from pydrake.all import RigidTransform, AbstractValue
import numpy as np


from typing import Dict

from utils import JointState

class NamesAvatarLeftFFOnly:
    def __init__(self) -> None:
        """ no J4, I removed it """
        
        self.names = [
            "lh_FFJ3",
            "lh_FFJ2",
            "lh_FFJ1",
        ]

class LowDOFRotateAction(LeafSystem):
    def __init__(self, 
                 ):
        LeafSystem.__init__(self)
        
        self.joint_action_size = 3
        
        # input port: action vector
        self.action_input = self.DeclareVectorInputPort(
            "action_input",
            BasicVector(self.joint_action_size)
        )
        
        # output port: joint action
        self.joint_action_output = self.DeclareAbstractOutputPort(
            "joint_action_output",
            lambda: AbstractValue.Make(JointState()),
            calc = self.calc_joint_action
        )
        
    def calc_joint_action(self, context, output):
        action_vec = self.action_input.Eval(context)
        
        assert(isinstance(action_vec, np.ndarray))
        joint_action = action_vec
        
        # squeeze out any extra dims
        joint_action = np.squeeze(joint_action)
        
        # put in a jointstate message so we can re-use the joint sub extractor
        joint_state_msg = JointState()
        joint_state_msg.position = joint_action
        
        joint_state_msg.name = NamesAvatarLeftFFOnly().names
        
        output.set_value(joint_state_msg)
        
        
    def wire_upstream(self, builder):
        pass
    
    
    
    

class LowDOFRotateObservation(LeafSystem):
    def __init__(self,
                ):
        LeafSystem.__init__(self)
        
        self.joint_action_size = 3
        
        self.DeclareAbstractInputPort(
            "state_saver_dict",
            AbstractValue.Make(Dict)
        )
        
        # Output Port: Observation dict
        self.DeclareAbstractOutputPort(
            "observation",
            lambda: AbstractValue.Make(dict()),
            self.CalcObservation
        )
        
    def CalcObservation(self, context, output):
        state_saver_dict = self.GetInputPort("state_saver_dict").Eval(context)
        assert(isinstance(state_saver_dict, dict))
        
        # just the joint state
        obs = state_saver_dict['joint_state']
        
        # assemble dict
        # out = {
        #     'joint_state': state_saver_dict['joint_state'],
        #     'target_wrench': state_saver_dict['target_wrench'],
        #     'current_wrench': state_saver_dict['current_wrench'],
        #     'manipulanda_pose': state_saver_dict['manipulanda_pose'],
        # }
        
        # set the vector output
        output.set_value(obs)
        
        
    def wire_upstream(self, builder, state_saver_output_port):
        builder.Connect(
            state_saver_output_port,
            self.GetInputPort("state_saver_dict")
        )