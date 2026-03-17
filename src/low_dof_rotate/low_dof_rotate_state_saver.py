









import time
import numpy as np
import pydrake.all
from pydrake.all import LeafSystem, AbstractValue
from pydrake.math import RigidTransform


import utils

from state_saver import StateSaver

from avatar import Avatar

        
class NamesAvatarLeftFFOnly:
    def __init__(self) -> None:
        """ no J4, I removed it """
        
        self.names = [
            "lh_FFJ3",
            "lh_FFJ2",
            "lh_FFJ1",
        ]

class TwoDOFDatasetSaverLeaf(StateSaver):
    def __init__(self, 
                 plant: pydrake.all.MultibodyPlant, 
                 avatar: Avatar,
                 target_body_index, 
                 manipulanda_body_index,
                 data_period = 0.1, 
                 file_path = "data/data.zarr",
                ):
        super().__init__(plant, target_body_index, data_period, file_path)
        
        self.plant = plant
        self.avatar = avatar
        self.target_body_index = target_body_index
        self.manipulanda_body_index = manipulanda_body_index
        
        # root body
        self.avatar_root_body_index = self.avatar.root_body.index()
        
        nb_dof = plant.num_positions(avatar.avatar_model_instance) + plant.num_velocities(avatar.avatar_model_instance)

        ## make the manip-anything specific ports
        # Robot state
        self.robot_state_port = self.DeclareVectorInputPort(
            "robot_state", nb_dof
        )
        
        # target wrench spatialforce
        self.target_wrench_port = self.DeclareAbstractInputPort(
            "target_wrench",
            AbstractValue.Make(pydrake.all.SpatialForce())
        )
        
        # current_wrench spatialforce
        self.current_wrench_port = self.DeclareAbstractInputPort(
            "current_wrench",
            AbstractValue.Make(pydrake.all.SpatialForce())
        )
        
        # reward value
        self.reward_port = self.DeclareVectorInputPort(
            "reward", 1
        )
        
        # save references to the fingertips, link names from avatar.urdf
        b2 = self.plant.GetBodyByName("lh_ffdistal") 
        
        self.lh_ff_tip_idx = b2.index()
        
        names_cls = NamesAvatarLeftFFOnly()
        self.names = names_cls.names
        
        self.create_ports()
        
        # analytics
        self.last_wall_time = time.time()
        
    def save_joint_positions(self, context):
        """
        save the current joint positions
        """
        # get robot state
        robot_state = self.robot_state_port.Eval(context)
        
        robot_state = np.array(robot_state)
        
        robot_positions = robot_state[:self.plant.num_positions(self.avatar.avatar_model_instance)]
        
        return robot_positions
        
    def robot_positions_map(self, positions):
        """
        all joint positions to our robot joint positions
        """                    
        out = []
        for name in self.names:
            joint = self.plant.GetJointByName(name, self.avatar.avatar_model_instance)
            idx = joint.position_start()
            out.append(positions[idx])
        
        return out
    
    def pos_from_plant(self, context, body_idx):
        """
        get the finger position from the pose port
        """
        poses = self.get_poses(context)
        
        pose: RigidTransform = poses[body_idx]
        
        pos = pose.translation()
        
        return pos
    
    def get_fk(self, context):
        
        # finger positions
        ff_pos = self.pos_from_plant(context, self.lh_ff_tip_idx)
        
        fk_state = np.concatenate([
            ff_pos,
        ])
        
        return fk_state
    
    def get_robot_joint_state(self, context):
        joint_positions = self.save_joint_positions(context)
        
        robot_positions = self.robot_positions_map(joint_positions)
        
        return robot_positions
    
    def get_pose_state(self, context):
        root_pose = self.get_avatar_root_pose(context)
        
        return root_pose
    
    def get_avatar_root_pose(self, context):
        """
        get the target pose of the object being manipulated
        """
        poses = self.get_poses(context)
        
        assert(isinstance(poses, list))
        pose = poses[self.avatar_root_body_index]
        
        # convert to numpy array
        pose_np = utils.pose_to_numpy(pose)
        
        return pose_np
    
    def get_a_object_pose(self, context, body_index):
        """
        get the pose of an object in the scene (not necessarily the target)
        """
        poses = self.get_poses(context)
        
        assert(isinstance(poses, list))
        object_pose = poses[body_index]
        
        # convert to numpy array
        object_pose_np = utils.pose_to_numpy(object_pose)
        
        return object_pose_np
    
    def get_object_target_pose(self, context):
        """
        get the target pose of the object being manipulated
        """
        return self.get_a_object_pose(context, self.target_body_index)
    
    def get_manipulanda_pose(self, context):
        """
        get the current pose of the object being manipulated
        """
        return self.get_a_object_pose(context, self.manipulanda_body_index)
    
    
    def get_rel_fk(self, fk_state, pose_state):
        """
        get the relative auxiliary state
        """
        # pos is the last 3
        avatar_root_pos = pose_state[4:7]
        
        # reshape to N x 3
        fk_reshaped = fk_state.reshape((-1, 3))
    
        rel_fk_state = fk_reshaped - avatar_root_pos
        
        # flatten
        rel_fk_state_flat = rel_fk_state.flatten()
        
        return rel_fk_state_flat
    
    def get_rel_object_target_pose(self, object_target_pose, pose_state):
        """
        get the relative object target pose
        """
        # convert numpy arrays back to RigidTransforms
        X_WA = utils.numpy_to_pose(pose_state)  # Avatar root in World
        X_WB = utils.numpy_to_pose(object_target_pose)  # Object target in

        # Calculate the transform of B relative to A
        X_AB = X_WA.inverse() @ X_WB
        
        # Convert to numpy array
        rel_object_target_pose = utils.pose_to_numpy(X_AB)
        
        return rel_object_target_pose
    
    def analytics_monitor(self, context):
        context_time = context.get_time()
        print(f"Assemble datapt Time: {context_time:.3f} s")
        
        # get the real time elapsed
        current_wall_time = time.time()
        
        # Calculate deltas
        wall_dt = current_wall_time - self.last_wall_time
        
        # print wall_dt
        print(f"Wall time delta: {wall_dt:.3f} s")
        
        self.last_wall_time = current_wall_time
        
    def assemble_datapt(self, context): #type:ignore
        """
        assemble the data point to be saved
        """
        if False:
            self.analytics_monitor(context)
        
        fk = self.get_fk(context)
                                
        object_target_pose = self.get_object_target_pose(context)
        
        manipulanda_pose = self.get_manipulanda_pose(context)
        
        pose_state = self.get_pose_state(context)
        
        joint_state = self.get_robot_joint_state(context)
        
        rel_fk = self.get_rel_fk(fk, pose_state)
        
        # hmmm ... during testing, reward is changing, but it's not exactly deg nor rads ...still works I guess
        reward = self.reward_port.Eval(context)[0] # type: ignore
        
        target_wrench: pydrake.all.SpatialForce = self.target_wrench_port.Eval(context) # type: ignore
        
        current_wrench: pydrake.all.SpatialForce = self.current_wrench_port.Eval(context) # type: ignore
                
        datapt = {
            'pose_state': pose_state,
            'joint_state': joint_state,
            'target_wrench': target_wrench.get_coeffs(),
            'current_wrench': current_wrench.get_coeffs(),
            'fk': fk,
            'rel_fk': rel_fk,
            # 'rel_object_pos': rel_object_pos,
            'object_id': 0, # from objects.yaml
            'object_target_pose': object_target_pose,
            'manipulanda_pose': manipulanda_pose,
            'not_done': True,
            'reward': reward,
            'task_id': 0, # no task id
            'qval': 0.0,
        }

        return datapt
    
    
    
    def wire_upstream(self, builder, reward_port, target_wrench_port, current_wrench_port): #type:ignore
        # bodies
        super().wire_upstream(builder)
        
        # plant robot state --> state saver robot state port
        builder.Connect(
            self.plant.get_state_output_port(self.avatar.avatar_model_instance),
            self.GetInputPort("robot_state")
        )
        
        # reward port
        builder.Connect(
            reward_port,
            self.GetInputPort("reward")
        )
        
        # target wrench port
        builder.Connect(
            target_wrench_port,
            self.GetInputPort("target_wrench")
        )
        
        # current wrench port
        builder.Connect(
            current_wrench_port,
            self.GetInputPort("current_wrench")
        )