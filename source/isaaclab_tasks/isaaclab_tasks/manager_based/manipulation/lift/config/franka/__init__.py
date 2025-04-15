# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os


from . import (
    agents,
    ik_abs_env_cfg,
    lift_ik_rel_env_cfg,
    lift_joint_pos_env_cfg,
    lift_ik_rel_blueprint_env_cfg,
    lift_joint_pos_instance_randomize_env_cfg,
    lift_ik_rel_instance_randomize_env_cfg
)


##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_joint_pos_env_cfg:FrankaCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Lift-Cube-Franka-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": lift_joint_pos_env_cfg.FrankaCubeLiftEnvCfg,
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-Lift-Cube-Instance-Randomize-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_joint_pos_instance_randomize_env_cfg.FrankaCubeLiftInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaCubeLiftEnvCfg",
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.ik_abs_env_cfg:FrankaTeddyBearLiftEnvCfg",
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_ik_rel_instance_randomize_env_cfg.FrankaCubeLiftInstanceRandomizeEnvCfg,
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_ik_rel_env_cfg.FrankaCubeLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     kwargs={
#         "env_cfg_entry_point": lift_ik_rel_env_cfg.FrankaCubeLiftEnvCfg,
#     },
#     disable_env_checker=True,
# )


# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Rel-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": lift_ik_rel_env_cfg.FrankaCubeLiftEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
#     },
#     disable_env_checker=True,
# )

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": ik_abs_env_cfg.FrankaCubeLiftEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-Franka-IK-Rel-Blueprint-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": lift_ik_rel_blueprint_env_cfg.FrankaCubeLiftBlueprintEnvCfg,
        # "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
    },
    disable_env_checker=True,
)

# gym.register(
#     id="Isaac-Lift-Cube-Franka-IK-Rel-Blueprint-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLMimicEnv",
#     kwargs={
#         "env_cfg_entry_point": lift_ik_rel_blueprint_env_cfg.FrankaCubeLiftBlueprintEnvCfg,
#         "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc.json"),
#     },
#     disable_env_checker=True,
# )




