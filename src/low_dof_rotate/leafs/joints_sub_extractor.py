

import numpy as np
from pydrake.all import LeafSystem, AbstractValue, BasicVector

from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
)

from utils import JointState

class JointSubExtractor(LeafSystem):
    def __init__(self, 
                 plant: MultibodyPlant,
                 robot_model_instance,
                 ):
        """
        Args:
            joint_names: List of strings in the exact order expected by the 
                         downstream Drake system/controller.
        """
        LeafSystem.__init__(self)
        self.plant = plant
        self.robot_model_instance = robot_model_instance
        
        
    def create_ports(self):
        self.num_actuators = self.plant.num_actuators(self.robot_model_instance)
        self.actuator_names = self.plant.GetActuatorNames(self.robot_model_instance)
        
        self.num_state_outputs = self.num_actuators * 2 # positions and velocities
        
                
        # 1. Input Port: ROS 2 JointState message
        self.DeclareAbstractInputPort(
            "joint_state_msg", 
            AbstractValue.Make(JointState())
        )
        
        # 2. Output Port: Vector of joint positions
        self.DeclareVectorOutputPort(
            "joint_positions",
            BasicVector(self.num_state_outputs),
            self.CalcJointPositions
        )

    def CalcJointPositions(self, context, output):
        # Retrieve the ROS message
        ros_msg = self.GetInputPort("joint_state_msg").Eval(context)
        
        assert(isinstance(ros_msg, JointState))
        
        # Create a mapping of Name -> Position from the incoming message
        # This handles cases where ROS sends more joints than we need
        # or sends them in a different order.
        ros_name_to_pos = {name: pos for name, pos in zip(ros_msg.name, ros_msg.position)}
        
        # Initialize output vector
        result = np.zeros(self.num_state_outputs) # positions and velocities
        
        # Fill the vector based on our requested order
        for i, target_name in enumerate(self.actuator_names):
            if target_name in ros_name_to_pos:
                result[i] = ros_name_to_pos[target_name]
                
        output.SetFromVector(result)
        
    def wire_upstream(self, builder, joint_state_msg_source_port):
        builder.Connect(
            joint_state_msg_source_port,
            self.GetInputPort("joint_state_msg")
        )   