# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sub-package with environment wrappers for Isaac Lab Mimic."""

import gymnasium as gym

from .franka_stack_ik_abs_mimic_env import FrankaCubeStackIKAbsMimicEnv
from .franka_stack_ik_abs_mimic_env_cfg import FrankaCubeStackIKAbsMimicEnvCfg
from .franka_stack_ik_rel_blueprint_mimic_env_cfg import FrankaCubeStackIKRelBlueprintMimicEnvCfg
from .franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv
from .franka_stack_ik_rel_mimic_env_cfg import FrankaCubeStackIKRelMimicEnvCfg
from .franka_stack_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeStackIKRelVisuomotorMimicEnvCfg
from .franka_lift_ik_rel_blueprint_mimic_env_cfg import FrankaCubeLiftIKRelBlueprintMimicEnvCfg
from .franka_lift_ik_rel_mimic_env import FrankaCubeLiftIKRelMimicEnv
from .franka_lift_ik_rel_mimic_env_cfg import FrankaCubeLiftIKRelMimicEnvCfg
from .franka_lift_ik_rel_visuomotor_mimic_env_cfg import FrankaCubeLiftIKRelVisuomotorMimicEnvCfg

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_mimic_env_cfg.FrankaCubeStackIKRelMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_blueprint_mimic_env_cfg.FrankaCubeStackIKRelBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKAbsMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_abs_mimic_env_cfg.FrankaCubeStackIKAbsMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_stack_ik_rel_visuomotor_mimic_env_cfg.FrankaCubeStackIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Rel-Mimic-v0",
#     entry_point="isaaclab_mimic.envs:FrankaCubeLiftIKRelMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": franka_lift_ik_rel_mimic_env_cfg.FrankaCubeLiftIKRelMimicEnvCfg,
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeLiftIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_lift_ik_rel_mimic_env_cfg.FrankaCubeLiftIKRelMimicEnvCfg,
        "robomimic_bc_cfg_entry_point": "/home/lab4/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/robomimic/bc.json",  # Adicione o caminho correto aqui
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-Blueprint-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeLiftIKRelMimicEnv", 
    kwargs={
        "env_cfg_entry_point": franka_lift_ik_rel_blueprint_mimic_env_cfg.FrankaCubeLiftIKRelBlueprintMimicEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0",
    entry_point="isaaclab_mimic.envs:FrankaCubeLiftIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": franka_lift_ik_rel_visuomotor_mimic_env_cfg.FrankaCubeLiftIKRelVisuomotorMimicEnvCfg,
    },
    disable_env_checker=True,
)