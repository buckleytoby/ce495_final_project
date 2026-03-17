import numpy as np
from pydrake.all import (
    LeafSystem, 
    Parser, 
    RigidTransform, 
    AbstractValue,
    ExternallyAppliedSpatialForce,
    SpatialForce,
    FramePoseVector,
    MultibodyPlant,
    SceneGraph,
    BodyIndex,
    ProximityProperties,
    SpatialInertia,
)

from pydrake.all import Box, RigidTransform



class ManipulandaLeaf(LeafSystem):
    """
    simple system to extract the manipulanda pose from the body poses input port
    """
    def __init__(self, 
                 plant: MultibodyPlant,
                 manipulanda_index: BodyIndex,
                 ):
        LeafSystem.__init__(self)
        self.plant = plant
        self.manipulanda_index = manipulanda_index
        
        
        
    def create_ports(self):
        # poses input port
        self.poses_port = self.DeclareAbstractInputPort(
            "body_poses", 
            self.plant.get_body_poses_output_port().Allocate()
        )
        
        # output: target pose
        self.DeclareAbstractOutputPort(
            "manipulanda_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcPose
        )

    def CalcPose(self, context, output):        
        poses = self.poses_port.Eval(context)
        
        assert(isinstance(poses, list))
        target_pose = poses[self.manipulanda_index]
        
        output.set_value(target_pose)
        
        
    
    def wire_upstream(self, builder):
        # Connect the input ports
        builder.Connect(
            self.plant.get_body_poses_output_port(),
            self.GetInputPort("body_poses")
        )
    
        
        