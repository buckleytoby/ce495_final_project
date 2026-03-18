
import logging
import yaml
import numpy as np
import csv
import os
from typing import List, Dict, Tuple, Optional
from pydrake.all import ModelInstanceIndex
from pydrake.multibody.tree import JointIndex, BodyIndex, FrameIndex
from pydrake.multibody.plant import MultibodyPlant, CoulombFriction
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.geometry import AddCompliantHydroelasticProperties, AddRigidHydroelasticProperties
from pydrake.multibody.plant import ContactModel, DiscreteContactApproximation
from pydrake.geometry import GeometrySet, CollisionFilterDeclaration
from pydrake.all import JointActuatorIndex, JointIndex, namedview
from pydrake.common.containers import namedview
from pydrake.multibody.tree import JointIndex
from pydrake.all import Quaternion

from pydrake.all import (
    Context,
)


def list_actuated_joints_from_kinematics(plant: MultibodyPlant):
    print(f"{'Joint Name':<20} | {'Type':<12} | {'Actuated?'}")
    print("-" * 45)
    
    # Iterate through all joints in the plant
    for i in range(plant.num_joints()):
        joint = plant.get_joint(JointIndex(i))
        
        # Check if the plant has an actuator for this specific joint index
        is_actuated = plant.HasJointActuatorNamed(joint.name())
        
        # Determine the joint type for clarity
        if joint.num_velocities() == 0:
            joint_type = "Fixed/Weld"
        elif joint.num_velocities() == 1:
            joint_type = "Revolute/Prism"
        else:
            joint_type = "Multi-DOF"

        # Only print if it's actually actuated
        if is_actuated:
            print(f"{joint.name():<20} | {joint_type:<12} | YES")
        else:
            print(f"{joint.name():<20} | {joint_type:<12} | NO")
            
            
            
def pose_to_numpy(pose: RigidTransform) -> np.ndarray:
    """
    assumes the vector order is wxyz, xyz
    """
    q = pose.rotation().ToQuaternion()
    p = pose.translation()
    
    # Pack row
    row = np.concatenate([q.wxyz(), p])
    
    return row

def numpy_to_pose(array: np.ndarray) -> RigidTransform:
    """
    assumes the vector order is wxyz, xyz
    """
    assert(array.ndim == 1)
    
    q = array[:4]
    p = array[4:7]
    
    quat = Quaternion(q[0], q[1], q[2], q[3])
    
    pose = RigidTransform(quat, p)
    
    return pose


    
def add_pose_to_vector(pose: RigidTransform, vector: np.ndarray):
    
    q = pose.rotation().ToQuaternion()
    p = pose.translation()
    
    # Pack row and append to Zarr
    row = np.concatenate([q.wxyz(), p])
    
    vector = np.concatenate([vector, row])
    
    return vector



class JointState:
    def __init__(self) -> None:
        self.position: list | np.ndarray = []
        self.name: list = []