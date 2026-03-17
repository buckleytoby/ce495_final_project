"""
drake leaf system which takes as input the object pose and the target pose and computes a binary reward: 1 if the object's position and orientation are within some threshold, epsilon of the target pose, else 0.
"""
from pydrake.math import RigidTransform
import numpy as np
from typing import Dict
from avatar_drake_sim.utils.avatar import Avatar
from avatar_drake_sim.utils import utils
from pydrake.all import MultibodyPlant
from pydrake.all import RigidTransform
from pydrake.all import LeafSystem, BasicVector, AbstractValue
from sensor_msgs.msg import JointState # ROS 2 message

# randomizer mixin
from avatar_drake_sim.leafs.randomizer_mixin import RandomizerMixin

from pydrake.all import (
    SpatialForce,
    ContactResults,
    BodyIndex,
    MultibodyPlant,
    DiagramBuilder,
)

class LowDOFRotateReward(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        
        # delayed signal input port, 1d vector
        self.DelayedJointPositionPort = self.DeclareVectorInputPort(
            "delayed_joint_position",
            BasicVector(1)
        )
        
        # current signal input port, 1d vector
        self.CurrentJointPositionPort = self.DeclareVectorInputPort(
            "current_joint_position",
            BasicVector(1)
        )
        
        # reward value output port, 1d vector
        self.RewardOutputPort = self.DeclareVectorOutputPort(
            "reward",
            BasicVector(1),
            self.CalcReward
        )
        
    def CalcReward(self, context, output):
        # get the delayed and current joint positions
        delayed_joint_position = self.DelayedJointPositionPort.Eval(context)[0] #type:ignore
        current_joint_position = self.CurrentJointPositionPort.Eval(context)[0] #type:ignore
        
        # compute the reward based on the change in joint position. Positive rotation = positive reward 
        reward = current_joint_position - delayed_joint_position
        
        # set the output reward value
        output.SetAtIndex(0, reward)
        
        
    def wire_upstream(self, builder, delayed_joint_position_port, current_joint_position_port):
        builder.Connect(
            delayed_joint_position_port,
            self.DelayedJointPositionPort
        )
        
        builder.Connect(
            current_joint_position_port,
            self.CurrentJointPositionPort
        )
        


class TargetWrenchReward(LeafSystem, RandomizerMixin):
    def __init__(self,
                 plant: MultibodyPlant,
                 body_index: BodyIndex,
                 ):
        LeafSystem.__init__(self)
        self.plant = plant
        self.body_index = body_index
        
        # target wrench parameter
        self.target_wrench = self.DeclareAbstractParameter(AbstractValue.Make(SpatialForce()))
        
        # current contacts input port
        self.contact_port = self.DeclareAbstractInputPort(
            "contact_results", 
            AbstractValue.Make(ContactResults())
        )
        
        # target wrench abstract output port
        self.target_wrench_output_port = self.DeclareAbstractOutputPort(
            "target_wrench",
            lambda: AbstractValue.Make(SpatialForce()),
            self.CalcTargetWrench
        )
        
        # current wrench abstract output port
        self.current_wrench_output_port = self.DeclareAbstractOutputPort(
            "current_wrench",
            lambda: AbstractValue.Make(SpatialForce()),
            self.CalcCurrentWrench
        )
        
        # reward value output port, 1d vector
        self.RewardOutputPort = self.DeclareVectorOutputPort(
            "reward",
            BasicVector(1),
            self.CalcReward
        )
        
    # def ConvertToWrench(self, info):
    #     # 1. Get the raw force and point
    #     f_Bc_W = info.contact_force()
    #     p_WC = info.contact_point()

    #     # 2. Point contacts have no local torque at the contact point
    #     # So we create a SpatialForce with zero torque.
    #     F_Bc_W = SpatialForce(tau=np.array([0, 0, 0]), f=f_Bc_W)
        
    #     # 3. Get the puck's current position (Center of Mass)
    #     # X_WB is the RigidTransform of the puck in the world
    #     X_WB = self.plant.EvalBodyPoseInWorld(context, plant.GetBodyByName("puck"))
    #     p_WBo = X_WB.translation()

    #     # 4. Calculate the offset vector from the Contact Point (C) to Body Origin (Bo)
    #     # Expressed in the World frame
    #     p_CBo_W = p_WBo - p_WC

    #     # 5. Shift the wrench! 
    #     # This automatically calculates the torque (r x f)
    #     F_Bo_W = F_Bc_W.Shift(p_CBo_W)

    #     # Now F_Bo_W.rotational() will give you the torque on the puck!
        
    def GetTotalContactWrench(self, context) -> SpatialForce:
        # --- 2. Contact Wrenches ---
        contact_results = self.contact_port.Eval(context)
        assert(isinstance(contact_results, ContactResults))
        
        total_wrench = SpatialForce.Zero()
                
        for i in range(contact_results.num_hydroelastic_contacts()):
            
            info = contact_results.hydroelastic_contact_info(i)
            
            total_wrench = info.F_Ac_W()
            
            # # Check if this body is part of the collision
            # if info.bodyA_index() == self.body_index:
            #     total_wrench -= info.contact_wrench().F_Cb_W
                
            # elif info.bodyB_index() == self.body_index:
            #     total_wrench += info.contact_wrench().F_Cb_W
                
        return total_wrench
    
    def CalcTargetWrench(self, context, output):
        target_wrench: SpatialForce = context.get_abstract_parameter(self.target_wrench).get_value()
        
        output.set_value(target_wrench)
        
    def CalcCurrentWrench(self, context, output):
        current_wrench: SpatialForce = self.GetTotalContactWrench(context)
        
        output.set_value(current_wrench)
        
    def CalcReward(self, context, output):
        # get the delayed and current joint positions
        current_wrench: SpatialForce = self.GetTotalContactWrench(context)
        
        # # split into force and torque?
        # current_force = current_wrench.translational()
        # current_torque = current_wrench.rotational()
        
        # get the target wrench from the parameter
        target_wrench: SpatialForce = context.get_abstract_parameter(self.target_wrench).get_value()
        
        # subtract the target wrench from the current wrench to get the error
        wrench_error = current_wrench - target_wrench
        
        # convert to vector
        # wrench_error_vec = wrench_error.get_coeffs()
        
        # for now only look at the z torque
        torque_error_vec = wrench_error.rotational()[2:3]
        
        # MSE it
        reward = -np.mean(torque_error_vec**2)
        
        # set the output reward value
        output.SetAtIndex(0, reward)
        
        
    def wire_upstream(self, builder: DiagramBuilder):
        builder.Connect(
            self.plant.get_contact_results_output_port(),
            self.contact_port
        )
        
    def randomize(self, simulator):
        # random z torque
        tauz = np.random.uniform(-0.5, 0.5)
        random_wrench = SpatialForce(tau=np.array([0, 0, tauz]), f=np.array([0, 0, 0]))
        
        root_context = simulator.get_mutable_context()
        
        reward_context = self.GetMyMutableContextFromRoot(root_context)
        
        reward_context.get_mutable_abstract_parameter(self.target_wrench).set_value(random_wrench)
