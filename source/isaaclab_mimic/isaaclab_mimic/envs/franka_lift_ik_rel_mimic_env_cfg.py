# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import h5py

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass


from isaaclab_tasks.manager_based.manipulation.lift.config.franka.lift_ik_rel_env_cfg import FrankaCubeLiftEnvCfg
# from isaaclab_mimic.envs.franka_lift_ik_rel_mimic_env import (
#     ajustar_subtarefas,
#     get_demo_length,  
#     validar_subtarefas,
#     calculate_dynamic_offsets,
#     validate_subtask_order,
# )

@configclass
class FrankaCubeLiftIKRelMimicEnvCfg(FrankaCubeLiftEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Lift Cube IK Rel env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # # Verificar se o dataset_path está configurado
        # if not hasattr(self.datagen_config, "dataset_path") or not self.datagen_config.dataset_path:
        #     import argparse
        #     parser = argparse.ArgumentParser()
        #     parser.add_argument("--input_file", type=str, required=True, help="Caminho para o dataset")
        #     args, _ = parser.parse_known_args()
        #     self.datagen_config.dataset_path = args.input_file

        # Override datagen configuration
        self.datagen_config.name = "demo_src_lift_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        # The following are the subtask configurations for the lift task.
        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.                
                object_ref="object",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).                
                subtask_term_signal="approach_obj",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.                
                subtask_term_offset_range=(0, 0),  
                # Selection strategy for the source subtask segment during data generation                
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="object",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="grasp_obj",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 0),  
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="object",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="lift_obj",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 10),  
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )

        # Levar o objeto para a posição alvo
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="object",
                # End of final subtask does not need to be detected
                subtask_term_signal=None,
                # No time offsets for the final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.03,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["franka"] = subtask_configs