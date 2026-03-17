import numpy as np
import pydrake.math
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.common import RandomGenerator
import random
from omegaconf import OmegaConf, DictConfig  # Import OmegaConf for YAML parsing

import pydrake.multibody.plant

from pydrake.all import (
    RevoluteJoint,
    SpatialVelocity,
    RigidBody,
    MultibodyPlant,
    Simulator,
)



class DomainRandomizer:
    def randomize_domain(self, simulator):
        raise NotImplementedError
    
    """ alias """
    def randomize(self, simulator):
        self.randomize_domain(simulator)
        
    def randomize_context(self, context):
        raise NotImplementedError

class RigidBodyRandomizer(DomainRandomizer):
    def __init__(self, 
                 plant: pydrake.multibody.plant.MultibodyPlant, 
                 body_ref: RigidBody,
                 pos_mins = [-0.25, 0.1, -0.25],
                 pos_maxs = [0.25, 0.5, 0.0],
                 randomize_orientation = True,
                 default_orientation = [1, 0, 0, 0],
                 ):
        self.plant = plant
        self.body_ref = body_ref
        self.pos_mins = pos_mins
        self.pos_maxs = pos_maxs
        self.randomize_orientation = randomize_orientation
        self.default_orientation = default_orientation
        
        # my members
        self.random_generator = RandomGenerator(random.randint(1, 999999))
        
    def random_rigid_body(self):
        
        # random sample
        random_position = np.random.uniform(self.pos_mins, self.pos_maxs) # Random x, y
        
        if self.randomize_orientation:
            q_raw = np.random.randn(4)
            q_unit = q_raw / np.linalg.norm(q_raw)
        else:
            q_unit = self.default_orientation

        # Create the Drake object
        drake_q = pydrake.common.eigen_geometry.Quaternion(q_unit) 

        # Convert to RotationMatrix for physics tasks
        R = RotationMatrix(drake_q)  
              
        pose = RigidTransform(R, random_position) #type:ignore
        
        return pose
    
    def randomize_domain(self, simulator: Simulator):
        root_context = simulator.get_mutable_context()
        
        self.randomize_context(root_context)
        
    def randomize_context(self, context):
        plant_context = self.plant.GetMyMutableContextFromRoot(context)    
            
        new_pose = self.random_rigid_body()
        
        # set new pose
        self.plant.SetFreeBodyPose(plant_context, self.body_ref, new_pose)
        
        # zero out velocity
        self.plant.SetFreeBodySpatialVelocity(self.body_ref,SpatialVelocity.Zero(), plant_context)
        
class TableRandomizer(RigidBodyRandomizer):
    def random_rigid_body(self):
        # for now, put params here
        x1 = -0.25
        y1 = -0.25
        z1 = 0.0
        yaw1 = -0.1
        
        x2 = 0.25
        y2 = 0.0
        z2 = 0.0
        yaw2 = 0.1
        
        # Generate random pose
        random_position = np.random.uniform([x1, y1, z1], [x2, y2, z2]) # Random x, y
        
        random_rotation = pydrake.math.RollPitchYaw(0, 0, np.random.uniform(yaw1, yaw2))  # Random yaw
        
        random_tf = RigidTransform(random_rotation, random_position)
        
        return random_tf
    
class RobotRandomizer(DomainRandomizer):
    def __init__(self, 
                 plant: MultibodyPlant, 
                 robot_joint_names,
                 joint_mins,
                 joint_maxs,
                 ):
        self.plant = plant
        self.robot_joint_names = robot_joint_names
        self.joint_mins = np.array(joint_mins)
        self.joint_maxs = np.array(joint_maxs)
        
        # my members
        self.random_generator = RandomGenerator(random.randint(1, 999999))
        
        
    def randomize(self, simulator: Simulator):
        """
        for each joint in robot joint names, get the joint limits, then sample a random position within those limits and set the joint to that position
        """
        root_context = simulator.get_mutable_context()
        plant_context = self.plant.GetMyMutableContextFromRoot(root_context)    
        
        for idx, joint_name in enumerate(self.robot_joint_names):
            joint = self.plant.GetJointByName(joint_name)
            
            assert(isinstance(joint, RevoluteJoint))
            
            # lower_limit = joint.position_lower_limits()[0]
            # upper_limit = joint.position_upper_limits()[0]
            lower_limit = self.joint_mins[idx]
            upper_limit = self.joint_maxs[idx]
            
            random_position = np.random.uniform(lower_limit, upper_limit)
            
            joint.set_angle(context=plant_context, angle=random_position)
    
        
class AvatarRandomizer(DomainRandomizer):
    def __init__(self, 
                 plant: pydrake.multibody.plant.MultibodyPlant, 
                 ):
        self.plant = plant
        
        # my members
        self.random_generator = RandomGenerator(random.randint(1, 999999))
        
    def random_avatar_joint_positions(self):
        # TODO: refactor to use joint.set_random_angle_distribution instead
        # Load joint limits from a YAML file
        joint_limits0 = OmegaConf.load("config/joint_limits.yaml")
        
        assert(isinstance(joint_limits0, DictConfig))
        joint_limits = joint_limits0["joint_limits"]
        
        joint_positions = {}
        for joint_key in joint_limits:
            joint = joint_limits[joint_key]
            lower = joint["min_position"]
            upper = joint["max_position"]
            
            rand_pos = np.random.uniform(lower, upper)
            
            joint_positions[joint_key] = rand_pos
        
        return joint_positions
    
    def randomize_domain(self, simulator):
        root_context = simulator.get_mutable_context()
        
        plant_context = self.plant.GetMyMutableContextFromRoot(root_context)    
            
        new_joint_positions = self.random_avatar_joint_positions()
        
        # set new joint positions
        for joint_key in new_joint_positions:
            joint = self.plant.GetJointByName(joint_key)
            angle = new_joint_positions[joint_key]
            
            assert(isinstance(joint, RevoluteJoint))
            joint.set_angle(context=plant_context, angle=angle)
