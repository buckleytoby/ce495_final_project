from pydrake.all import (
    MultibodyPlant,
    PdControllerGains,
    RigidTransform,
)

import omegaconf
from omegaconf import OmegaConf

import numpy as np

import pydrake
import pydrake.all
import pydrake.math


from domain_randomizer import AvatarRandomizer, TableRandomizer

# import randomizer_mixin
from leafs.randomizer_mixin import RandomizerMixin

class AddModels:
    def __init__(self,
                 plant,
                 parser):
        self.plant = plant
        self.parser = parser

class AddModelsAvatar(AddModels):
    def add_models(self):
        self.avatar_model_instance = self.parser.AddModels("urdf/avatar.urdf")[0] #type:ignore
        
        # string from the urdf
        self.root_body = self.plant.GetBodyByName("avatar_mount_shopstand", self.avatar_model_instance)
        
class AddModelsAvatarLeftHand(AddModels):
    def add_models(self):
        # full path to urdf
        avatar_path = root + "/urdf/avatar_left_hand.urdf" #type:ignore
        
        self.avatar_model_instance = self.parser.AddModels(avatar_path)[0] #type:ignore
        
        # string from the urdf
        self.root_body = self.plant.GetBodyByName("lh_forearm", self.avatar_model_instance)
        
class AddModelsAvatarLeftFFOnly(AddModels):
    def add_models(self):
        # full path to urdf
        avatar_path = "./urdf/avatar_left_ff_only.urdf" #type:ignore
        
        self.avatar_model_instance = self.parser.AddModels(avatar_path)[0] #type:ignore
        
        # string from the urdf
        self.root_body = self.plant.GetBodyByName("lh_ffknuckle", self.avatar_model_instance)

class Avatar(RandomizerMixin):
    def __init__(self,
                 plant: MultibodyPlant,
                 scene_graph: pydrake.all.SceneGraph | None,
                 parser: pydrake.all.Parser
                 ):
        self.plant = plant
        self.scene_graph = scene_graph
        self.parser = parser
        
        self.add_models()
        
        if False:
            self.disable_collisions()
        
        # randomizer
        self.plant_randomizer = AvatarRandomizer(plant = self.plant)
        
        self.table_randomizer = TableRandomizer(
            plant = self.plant,
            body_ref = self.root_body
        )
    
    def add_models(self):
        if False:
            pass
        
        # avatar left hand
        if True:
            add_model = AddModelsAvatarLeftHand(self.plant, self.parser)
            add_model.add_models()
            
            self.avatar_model_instance = add_model.avatar_model_instance
            self.root_body = add_model.root_body
        
    def disable_collisions(self):
        """
        Disable collisions for the mount body
        """
        if self.scene_graph is None:
            return
            
        for geometry_id in self.plant.GetCollisionGeometriesForBody(self.root_body):
            self.scene_graph.RemoveRole(self.plant.get_source_id(), geometry_id, pydrake.all.Role.kProximity)
        
    def weld_root(self):
        """
        only use if you don't need the mount to move after the sim begins
        """
        assert(not self.plant.is_finalized())
        
        # Use SetFreeBodyPose if it's not welded, or SetDefaultFreeBodyPose before Finalize
        tf = pydrake.all.RigidTransform([1.0, 0.0, 0.0]) # type:ignore
        
        self.plant.WeldFrames(self.plant.world_frame(), self.root_body.body_frame(), tf)

    def randomize_domain(self, simulator: pydrake.all.Simulator):
        root_context = simulator.get_mutable_context()
        
        plant_context = self.plant.GetMyMutableContextFromRoot(root_context)    
            
        # mount location
        self.table_randomizer.randomize_domain(simulator)
        
        # avatar joint position
        if False: # only for full avatar
            self.plant_randomizer.randomize_domain(simulator)
            
    def set_plant_gains(self):
        # get the robot's idx
        robot_model_idx = self.avatar_model_instance
        
        # full path to urdf
        path = "./config/controller_gains.yaml" #type:ignore
        
        
        # load the controller gain yaml using omegaconf
        config = OmegaConf.load(path)

        assert(isinstance(config, omegaconf.DictConfig))
        controller_gains = config["controller_gains"]["implicit_pd"]
        
        # set actuator gains for implicit PD control. Actuation will now be solved in the plant
        for actuator_name, gains in controller_gains.items():
            # existence assertion
            # assert(self.plant.HasJointActuatorNamed(actuator_name, robot_model_idx)), f"Actuator '{actuator_name}' not found in robot model"
            
            if not self.plant.HasJointActuatorNamed(actuator_name, robot_model_idx):
                print(f"Warning: Actuator '{actuator_name}' not found in robot model. Skipping gain setting for this actuator.")
                continue

            # get the actuator
            actuator = self.plant.GetJointActuatorByName(actuator_name, robot_model_idx)
            
            # create the pd gain object
            pd_gains = PdControllerGains(p=gains["kp"], d=gains["kd"])
            
            # set it
            actuator.set_controller_gains(pd_gains)
            
            
class AvatarLeftFFOnly(Avatar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # weld the root
        self.weld_root()
        
    def add_models(self):
        add_model = AddModelsAvatarLeftFFOnly(self.plant, self.parser)
        add_model.add_models()
        
        self.avatar_model_instance = add_model.avatar_model_instance
        self.root_body = add_model.root_body
        
    def weld_root(self):
        """
        only use if you don't need the mount to move after the sim begins
        
        weld to xyz 0, 0, 0, rpy 90 deg, 0, 0
        """
        assert(not self.plant.is_finalized())
        
        rt = RigidTransform(pydrake.math.RollPitchYaw(np.deg2rad([-90, +90, 0])), np.array([0.0, 0.0, 0.0]))
        
        self.plant.WeldFrames(self.plant.world_frame(), self.root_body.body_frame(), rt)