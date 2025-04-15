# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass


from isaaclab_tasks.manager_based.manipulation.lift.config.franka.lift_ik_rel_env_cfg import FrankaCubeLiftEnvCfg

# from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg as FrankaCubeLiftEnvCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
# from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.assets import RigidObjectCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
# from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
# from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


@configclass
class FrankaCubeLiftIKRelMimicEnvCfg(FrankaCubeLiftEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Lift Cube IK Rel env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override datagen configuration
        self.datagen_config.name = "demo_src_lift_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Subtask configurations
        subtask_configs = []

        # Reaching object
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="approach_obj",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Grasping and lifting 5 cm
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="grasp_obj",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # Lifting object
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="lift_obj",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )
        # Lifting to target object position
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal="target_object_position",  # Alterado de "lift_obj" para "target_object_position"
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        self.subtask_configs["franka"] = subtask_configs
