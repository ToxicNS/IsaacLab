# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_tasks.manager_based.manipulation.lift.mdp import franka_lift_events
from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    # # camera
    # camera = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Camera",
    #     #collision_props=sim_utils.CollisionPropertiesCfg(),  # Habilita a colisão sem corpo rígido
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[1.0, 0.0, 0.5], rot=[0.0, -0.18, 0.0, 0.707]),
    #     spawn=UsdFileCfg(usd_path="file:///home/lab4/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/objets/rsd455.usd"),
    # )


##
# MDP settings
##

@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.5, 0.5), pos_y=(-0.25, -0.25), pos_z=(0.1, 0.1), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    # arm_action: mdp.JointPositionActionCfg = MISSING
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations para o grupo de políticas."""

        actions = ObsTerm(func=mdp.last_action)        

        # Adicionar novas observações
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Observações relacionadas ao objeto
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        
        # Observações existentes que você deseja manter
        object_positions = ObsTerm(func=mdp.object_positions_in_world_frame)
        object_orientations = ObsTerm(func=mdp.object_orientations_in_world_frame)
        instance_randomized_object_positions = ObsTerm(func=mdp.instance_randomize_object_positions_in_world_frame)
        instance_randomized_object_orientations = ObsTerm(func=mdp.instance_randomize_object_orientations_in_world_frame)
        
        # Observações relacionadas ao end-effector
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        
        # Observações relacionadas ao gripper
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        
        # Outras observações de objeto
        object_obs = ObsTerm(func=mdp.object_obs)
        instance_randomized_object_obs = ObsTerm(func=mdp.instance_randomize_object_obs)

        def __post_init__(self):
            """Configurações adicionais."""
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class RGBCameraPolicyCfg(ObsGroup):
        """Observations for policy group with RGB images."""

        # table_cam = ObsTerm(
        #     func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False}
        # )
        # wrist_cam = ObsTerm(
        #     func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False}
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observações para o grupo de subtarefas."""

        # Verifica se o end-effector está próximo do objeto
        approach_obj = ObsTerm(
            func=mdp.object_reached_goal,
            params={
                "command_name": "object_pose",  
                "threshold": 0.05,  
                "robot_cfg": SceneEntityCfg("robot"),  
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # Verifica se o objeto foi agarrado
        grasp_obj = ObsTerm(
            func=mdp.object_grasped,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
                "grasp_distance": 0.02,
            },
        )

        # Verifica se o objeto foi levantado acima de uma altura mínima
        lift_obj = ObsTerm(
            func=mdp.object_is_lifted,
            params={
                "minimal_height": 0.05,
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        # Substituir stacked_obj por target_object_position
        target_object_position = ObsTerm(
            func=mdp.object_reached_goal,
            params={
                "command_name": "object_pose",
                "threshold": 0.05, 
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            """Configurações adicionais."""
            self.enable_corruption = False
            self.concatenate_terms = False
    

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg() 


@configclass
class EventCfg:
    """Unified configuration for simulation events."""

    # Reset full scene
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset"
    )

    # Reset object position to a fixed pose
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.5, 0.5),
                "y": (0.20, 0.20),
                "z": (0.02, 0.02),
                "yaw": (-1.0, 1.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Set default arm joint pose
    init_franka_arm_pose = EventTerm(
        func=franka_lift_events.set_default_joint_pose,
        mode="startup",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )

    # Randomize arm joint states (Gaussian noise)
    randomize_franka_joint_state = EventTerm(
        func=franka_lift_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Randomize object pose (only yaw changes)
    randomize_object_pose = EventTerm(
        func=franka_lift_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {"x": (0.5, 0.5), "y": (0.20, 0.20), "z": (0.0203, 0.0203), "yaw": (0, 0, 0)},
            "min_separation": 0.1,
            "asset_cfgs": [SceneEntityCfg("object")],
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # object_reached_goal = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    object_reached_goal = RewTerm(
        func=mdp.object_goal_distance_sparse,
        params={
            "threshold": 0.05, 
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object")
        }, 
        weight=5.0  # Aumentado para maior impacto
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.02},  # Reduzido para facilitar o aprendizado inicial
        weight=20.0  # Aumentado para maior impacto
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=20.0  # Aumentado para maior impacto
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=10.0  # Aumentado para maior impacto
    )

    approach_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object")},
        weight=10.0  # Nova recompensa para incentivar aproximação inicial
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)  # Penalidade ajustada

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-2,  # Penalidade ajustada
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

        
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Condição de término por tempo limite
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Condição de término se o objeto cair
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(
        func=mdp.object_reached_goal,
        params={
            "command_name": "object_pose",
            "threshold": 0.05,
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )

    # success = DoneTerm(func=mdp.object_reached_goal)    

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )  


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()  
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()  

    events: EventCfg = EventCfg() 
    curriculum: CurriculumCfg = CurriculumCfg() 

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2 
        self.episode_length_s = 30.0
        # Adicionar seed para o ambiente base
        self.seed = 1  # Adicione esta linha        
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625