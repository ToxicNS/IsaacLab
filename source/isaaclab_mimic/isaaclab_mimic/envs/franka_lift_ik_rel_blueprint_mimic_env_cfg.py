# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.lift_ik_rel_blueprint_env_cfg import (
    FrankaCubeLiftBlueprintEnvCfg,
)


@configclass
class FrankaCubeLiftIKRelBlueprintMimicEnvCfg(FrankaCubeLiftBlueprintEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Lift Cube IK Rel env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()

        # Override the existing values
        self.datagen_config.name = "isaac_lab_franka_lift_ik_rel_blueprint_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # The following are the subtask configurations for the stack task.
        subtask_configs = []
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

        # Subtarefa: Agarrar o objeto
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

        # Subtarefa: Levantar o objeto
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

        self.subtask_configs["franka"] = subtask_configs
