


"""

uses only the avatar left ff

"""





import time
import pydrake
import numpy as np


import pydrake.all
import pydrake.visualization
from pydrake.all import (
    DiagramBuilder,
    AddMultibodyPlantSceneGraph, 
    Parser, 
    Simulator,
    TrajectorySource,
    PidController,
    StartMeshcat,
    ModelInstanceIndex,
    PdControllerGains,
    MultibodyPlant,
    RigidTransform,
    DiscreteTimeDelay,
    LeafSystem,
    BasicVector,
    )

import pydrake.geometry
from pydrake.geometry import (
    RenderEngineVtkParams, 
    MakeRenderEngineVtk
)
import pydrake.multibody
import pydrake.multibody.plant

from leafs import (
    meshcat_keyboard,
    episode_monitor,
    joints_sub_extractor,
    manipulanda,
)

from pydrake.systems.controllers import (
    InverseDynamics,
)


from avatar import AvatarLeftFFOnly

from low_dof_rotate_reward import LowDOFRotateReward, TargetWrenchReward

import utils

# suppress avatar urdf warnings
import logging; logging.getLogger("drake").setLevel(logging.ERROR)

from sim_base import Simulation

from pydrake.all import EventStatus

import time

from low_dof_rotate_state_saver import TwoDOFDatasetSaverLeaf

class NamesAvatarLeftFFOnly:
    def __init__(self) -> None:
        """ no J4, I removed it """
        
        self.names = [
            "lh_FFJ3",
            "lh_FFJ2",
            "lh_FFJ1",
        ]

class GetJointRotation(LeafSystem):
    def __init__(self, plant, joint_name):
        LeafSystem.__init__(self)
        self.plant = plant
        # Find the specific index where this joint's position starts in the state vector
        self.joint = plant.GetJointByName(joint_name)
        
        
    def create_ports(self):
        # 1. Input Port: The full state of the MultibodyPlant
        self.DeclareVectorInputPort(
            "plant_state", self.plant.num_multibody_states()
        )
        
        # 2. Output Port: Just the single joint value
        self.DeclareVectorOutputPort(
            "joint_value", BasicVector(1), self.CalcJointValue)

    def CalcJointValue(self, context, output):
        # Evaluate the input port to get the full state vector
        state = self.get_input_port(0).Eval(context)
        assert(isinstance(state, np.ndarray))
        
        # Safe way: slice the input vector using the joint's position index
        # This works even if the input is coming from a non-plant system
        theta = state[self.joint.position_start()]
        
        output.SetAtIndex(0, theta)
        
    def wire_upstream(self, builder):
        builder.Connect(
            self.plant.get_state_output_port(), 
            self.GetInputPort("plant_state")
        )
        
class LowDOFRotateSim(Simulation):
    def __init__(self,
                 must_connect_ros2 = True,
                 ):
        super().__init__()
        self.must_connect_ros2 = must_connect_ros2
        
        # my params
        self.dt = 0.00025
        self.duration = np.inf
        
        # data sampling rate
        self.data_sampling_period = 0.1
        
        # my members
        self.builder = DiagramBuilder()
        
    def create_ports(self):
        self.manipulanda_leaf.create_ports()
        
        self.joint_sub_extractor_leaf.create_ports()
        
        self.puck_joint_leaf.create_ports()
        
    def setup_meshcat(self):
        self.meshcat: pydrake.geometry.Meshcat = StartMeshcat()
        
        # Visualization settings
        vis_config = pydrake.visualization.VisualizationConfig()
        
        # publish_contacts = False: Don't visualize contact forces
        vis_config.publish_contacts = False
        
        # delete_on_initialization_event = True: Geometry visualization reset on reset
        vis_config.delete_on_initialization_event = False
        
        vtk_params = RenderEngineVtkParams()
        vtk_params.cast_shadows = True
        render_engine = MakeRenderEngineVtk(vtk_params)

        renderer_name = "vtk_renderer"
        self.scene_graph.AddRenderer(renderer_name, render_engine)     
        
        self.meshcat.SetProperty("/visualizer", "castShadow", True)
        self.meshcat.SetProperty("/visualizer", "receiveShadow", True)

        # apply
        pydrake.visualization.ApplyVisualizationConfig(vis_config, self.builder, scene_graph = self.scene_graph, meshcat = self.meshcat)   
        
        # telemetry
        # # Set initial text
        # self.meshcat.SetProperty("/Grid", "visible", True) # Just to ensure meshcat is active
        # self.meshcat.set(
        #     "<strong>Stats</strong><br>Reward: 0.0<br>Target: [0, 0, 10]", 
        #     id="telemetry"
        # )

        # # To update it during your simulation/reset loop:
        # self.meshcat.Set2dDisplayHtml(
        #     f"<strong>Stats</strong><br>Reward: {reward_val:.2f}", 
        #     id="telemetry"
        # )
    
    def reset(self):
        """
        Reset all bodies to initial positions, and initialize simulation context.
        """    
        # Get context
        sim_context = self.simulator.get_mutable_context()
        context = self.plant.GetMyContextFromRoot(sim_context)
                
        # Reset simulation
        self.plant.SetDefaultContext(context)
        
        # domain randomization
        self.randomize()
            
        # reset time
        sim_context.SetTime(0)
        self.simulator.Initialize()
        
        # reset the state saver
        self.state_saver.reset()
    
        
    def add_objects(self):
        # full path to urdf
        path = "/urdf/rotating_puck.sdf" #type:ignore
        
        self.manipulanda_model_instance = self.parser.AddModels(path)[0] #type:ignore
        
    def init_and_weld_objects(self):
        # weld it to xyz [0.1, 0.1, 0.0]
        rt = RigidTransform(np.array([0.05, 0.05, 0.0]))
        
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("base_fixed", self.manipulanda_model_instance),
            rt,
        )
        
    def setup_force_mux(self):
        """
        separate out the muxer because I can predict that this'll grow in the future
        """
        pass

    def setup_plant(self):
        # typing
        self.plant: pydrake.multibody.plant.MultibodyPlant
        
        # init plant and scene graph
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, self.dt)
        
        # init parser
        self.parser = Parser(self.plant)
        
        # add avatar
        self.avatar = AvatarLeftFFOnly(self.plant, self.scene_graph, self.parser)
        
        # add objects
        self.add_objects()
        
        # init and weld
        self.init_and_weld_objects()
        
        # no gravity
        self.plant.mutable_gravity_field().set_gravity_vector(np.array([0, 0, 0]))
        
        # default friction
        default_friction = pydrake.multibody.plant.CoulombFriction(static_friction=0.6, dynamic_friction=0.5)
        
        # contact type
        self.plant.set_contact_model(pydrake.multibody.plant.ContactModel.kHydroelasticWithFallback)
        
        utils.list_actuated_joints_from_kinematics(self.plant)
        
        
        
    def add_controller_plant(self):
        """
        add the dummy controller plant, used for gravity compensation
        """
        
        plant = MultibodyPlant(self.dt)
        self.controller_plant = self.builder.AddSystem(
            plant
        )
        self.controller_plant.set_name("controller_plant")
        
        # temp parser
        parser = Parser(plant)
        
        # add the controller avatar
        self.controller_avatar = AvatarLeftFFOnly(plant, None, parser)
        
        plant.Finalize()
            
    def add_gravity_comp(self):
        """
        need a dummy plant to be used in the inverse dynamics system
        """
        controller_context = self.controller_plant.CreateDefaultContext()
        
        # create an inverse dynamics system which will be used just for gravity compensation
        id_system = InverseDynamics(
            self.controller_plant, #type:ignore
            InverseDynamics.kGravityCompensation, 
            controller_context
        )
        
        # add gravity comp
        self.gravity_comp_inverse_dynamics = self.builder.AddSystem(id_system)
        
        # set the name
        self.gravity_comp_inverse_dynamics.set_name("gravity_comp_inverse_dynamics")
        
    def add_controller(self):
        # dummy controller plant
        self.add_controller_plant()
        
        # plant gains
        self.avatar.set_plant_gains()
        
        self.add_gravity_comp()
        
    def make_leafs(self):
        """
        makes and registers leafs that need randomization
        """
        # for now just use one object
        index = self.plant.GetBodyIndices(self.manipulanda_model_instance)[0]
        
        # manipulanda pose extractor
        self.manipulanda_leaf = manipulanda.ManipulandaLeaf(
            self.plant,
            index,
        )
        
        # process joint cmds into a state usable by the plant
        self.joint_sub_extractor_leaf = joints_sub_extractor.JointSubExtractor(
            self.plant,
            self.avatar.avatar_model_instance,
        )
        
        # puck joint leaf
        self.puck_joint_leaf = GetJointRotation(
            self.plant,
            "puck_revolute_joint"
        )
        
        # time delay for reward signal
        self.reward_delay = DiscreteTimeDelay(update_sec = self.data_sampling_period, delay_time_steps = 1, vector_size = 1)
        
        # reward leaf
        self.reward_leaf = LowDOFRotateReward()
        self.reward_wrench_leaf = TargetWrenchReward(
            self.plant,
            index,
        )
        
        
        ######## REQUIRED FOR RANDOMIZATION ########
        self.register_system(self.reward_wrench_leaf)
        
        
    def add_leafs(self):
        """
        add all leafs before wiring any, since wiring might depend on any arbitrary system
        """
        
        # avatar controller
        self.add_controller()
        
        ### common
        # keyboard I/O
        self.keyboard = self.builder.AddSystem(
            meshcat_keyboard.MeshcatKeyboardReader(self.meshcat)
        )
        
        # episode reset monitor
        self.ep_reset_trigger = self.builder.AddSystem(
            episode_monitor.KeyTrigger()
        )
        
        # keyboard save monitor
        self.ep_save_trigger = self.builder.AddSystem(
            episode_monitor.KeyTrigger()
        )
        
        # for now just use one body
        index = self.plant.GetBodyIndices(self.manipulanda_model_instance)[0]
        
        # state saver        
        self.state_saver = TwoDOFDatasetSaverLeaf(
                self.plant, 
                self.avatar,
                index,
                manipulanda_body_index = index,
                file_path = "data/two_dof_rotate_sim.zarr",
                data_period = self.data_sampling_period, # / 3.0, # nominal is 10hz, but the sim takes 3x real time, so divide by 3?
            )
        
        self.builder.AddSystem(self.state_saver)
        self.builder.AddSystem(self.manipulanda_leaf)
        self.builder.AddSystem(self.joint_sub_extractor_leaf)
        self.builder.AddSystem(self.puck_joint_leaf)
        self.builder.AddSystem(self.reward_delay)
        self.builder.AddSystem(self.reward_leaf)
        self.builder.AddSystem(self.reward_wrench_leaf)
        
        
    def num_actions(self):
        # hard coded for now...
        return 3

    def wire_leafs(self):
        """
        wire up all leafs
        """
        robot_model_idx = self.avatar.avatar_model_instance
        
        # implicit pd, joint commands --> plant desired state
        self.builder.Connect(
            self.joint_sub_extractor_leaf.GetOutputPort("joint_positions"),
            self.plant.get_desired_state_input_port(robot_model_idx)
        )
        
        # plant state --> gravity comp system
        self.builder.Connect(
            self.plant.get_state_output_port(robot_model_idx), 
            self.gravity_comp_inverse_dynamics.get_input_port_estimated_state() #type:ignore
        )
        
        ### common
        # keyboard --> episode reset trigger
        self.builder.Connect(
            self.keyboard.GetOutputPort("reset_episode"),
            self.ep_reset_trigger.get_input_port()
        )
        
        # keyboard --> episode save trigger
        self.builder.Connect(
            self.keyboard.GetOutputPort("save_episode"),
            self.ep_save_trigger.get_input_port()
        )
        
        
        ### state saver
        self.state_saver.wire_upstream(self.builder, self.reward_leaf.GetOutputPort("reward"), self.reward_wrench_leaf.GetOutputPort("target_wrench"), self.reward_wrench_leaf.GetOutputPort("current_wrench"))
        
        self.manipulanda_leaf.wire_upstream(self.builder)
        
        # puck joint leaf
        self.puck_joint_leaf.wire_upstream(self.builder)
        
        # reward delay
        self.builder.Connect(
            self.puck_joint_leaf.GetOutputPort("joint_value"),
            self.reward_delay.get_input_port()
        )
        
        # reward leaf
        self.reward_leaf.wire_upstream(self.builder, self.reward_delay.get_output_port(), self.puck_joint_leaf.GetOutputPort("joint_value"))
        self.reward_wrench_leaf.wire_upstream(self.builder)
        
    def randomize(self):
        for randomizer in self.randomizers:
            randomizer.randomize(self.simulator)
            
        # # special, for the avatar
        # self.avatar.randomize_domain(self.simulator)
        pass
        
        
    def setup_randomizers(self):
        pass
        
    def flush_buffers(self, root_context):
        context = self.state_saver.GetMyContextFromRoot(root_context)
        
        # see if 
        
        # force an eval so the leaf is fresh
        # self.state_saver.GetInputPort("trigger").Eval(context)
        
        # # force a publish
        # self.state_saver.ForcedPublish(context)
        
        self.state_saver.GetOutputPort("flusher").Eval(context)
        
    def episode_end(self, root_context):
        # flush buffers        
        # self.flush_buffers(root_context) # handled by pressing 's' now (before ending an episode)
        
        # reset ep
        self.reset()
        
    def calc_must_reset(self, root_context):
        # check for reset button
        reset_triggered = self.ep_reset_monitor(root_context)
        
        must_reset = reset_triggered 
        
        return must_reset
    
    def calc_must_save(self, root_context):
        # check for save button
        save_triggered = self.ep_save_monitor(root_context)
        
        must_save = save_triggered
        
        return must_save
    
    def sim_monitor(self, root_context):
        
        # normal status
        event_status = EventStatus.Succeeded()
        
        must_reset = self.calc_must_reset(root_context)
        must_save = self.calc_must_save(root_context)
         
        # first save
        if must_save:
            print("Saving episode...")
            self.flush_buffers(root_context)
            print("Episode saved.")
            
        # then reset
        if must_reset:
            print("Resetting episode...")
            self.episode_end(root_context)
            print("Episode reset.")
            
        return event_status
            
        
        
    def ep_reset_monitor(self, root_context):
        context = self.ep_reset_trigger.GetMyContextFromRoot(root_context)
        
        is_episode_done = self.ep_reset_trigger.get_output_port(0).Eval(context)
    
        return is_episode_done
            
    def ep_save_monitor(self, root_context):
        context = self.ep_save_trigger.GetMyContextFromRoot(root_context)
        
        is_save_triggered = self.ep_save_trigger.get_output_port(0).Eval(context)
    
        return is_save_triggered
            
    def setup_simulator(self):
        # compile diagram
        self.diagram = self.builder.Build()
        
        
        # setup sim
        self.simulator = Simulator(self.diagram)
        self.simulator.set_target_realtime_rate(1.0)
        self.simulator.Initialize()
        
        # setup episode end monitor, also terminates episodes for drake-gym
        self.simulator.set_monitor(self.sim_monitor)

    def run(self):
        
        # initial reset
        self.reset()
        
        # run it
        self.simulator.AdvanceTo(self.duration)

def setup_sim(must_connect_ros2 = True):
    # 1. Initialize the simulation object
    sim = LowDOFRotateSim(must_connect_ros2)
    
    # setup the plant
    sim.setup_plant()
    
    # make leafs
    sim.make_leafs()
    
    ## finalize plant
    sim.plant.Finalize()
    
    # some ports depend on a finalized plant, so must be called after Finalize
    sim.create_ports()
    
    # meshcat
    sim.setup_meshcat()
    
    # add all leafs
    sim.add_leafs()
    
    # wire leafs
    sim.wire_leafs()
    
    # mux 
    sim.setup_force_mux()
    
    # setup sim
    sim.setup_simulator()
    
    # randomizer
    sim.setup_randomizers()
    
    return sim

def main():
    sim = setup_sim()
    
    # start the simulation
    print("Starting simulation (Gravity=0, Air Resistance=On)...")
    print("don't forget to run the haptx driver and the avatar driver (alias: start_drake_run_avatar)")
    sim.run()
    print("Simulation complete.")

if __name__ == "__main__":
    main()