# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==========================
# Robot State Observations
# ==========================

# def joint_pos_rel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Retorna as posições relativas das juntas do robô."""
#     robot: Articulation = env.scene[robot_cfg.name]
    
#     # Pegamos todas as juntas exceto as duas do gripper (que são as últimas)
#     joint_pos = robot.data.joint_pos[:, :-2]
    
#     # Normaliza usando os limites das juntas
#     joint_min = robot.data.joint_pos_limits[:, :-2, 0]  # Atualizado
#     joint_max = robot.data.joint_pos_limits[:, :-2, 1]  # Atualizado
#     joint_range = joint_max - joint_min
    
#     # Evita divisão por zero (juntas que não têm range)
#     joint_range = torch.where(joint_range == 0, torch.ones_like(joint_range), joint_range)
    
#     # Normalização para o intervalo [-1, 1]
#     joint_pos_normalized = 2.0 * (joint_pos - joint_min) / joint_range - 1.0
    
#     return joint_pos_normalized


# def joint_vel_rel(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
#     """Retorna as velocidades relativas das juntas do robô."""
#     robot: Articulation = env.scene[robot_cfg.name]
    
#     # Pegamos todas as juntas exceto as duas do gripper (que são as últimas)
#     joint_vel = robot.data.joint_vel[:, :-2]
    
#     # Assumindo que as velocidades estão no intervalo [-max_speed, max_speed]
#     max_speed = 5.0  # valor típico, ajuste conforme necessário
    
#     # Normalização para o intervalo [-1, 1]
#     joint_vel_normalized = torch.clamp(joint_vel / max_speed, -1.0, 1.0)
    
#     return joint_vel_normalized

def joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Retorna as posições relativas das juntas do robô."""
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Retorna as velocidades relativas das juntas do robô."""
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_vel

# ==========================
# Command Observations
# ==========================

def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Retorna os comandos gerados pelo gerenciador de comandos."""
    return env.command_manager.get_command(command_name)

# ==========================
# Object State Observations
# ==========================

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Retorna a posição do objeto no frame raiz do robô.

    Args:
        env: O ambiente.
        robot_cfg: Configuração do robô.
        object_cfg: Configuração do objeto.

    Returns:
        torch.Tensor: Posição do objeto no frame raiz do robô.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a posição do objeto no frame do mundo."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w


def instance_randomize_object_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a posição do objeto no frame do mundo com randomização de instâncias."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    object: RigidObjectCollection = env.scene[object_cfg.name]
    object_pos_w = []
    for env_id in range(env.num_envs):
        object_pos_w.append(object.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
    return torch.stack(object_pos_w)


def object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a orientação do objeto no frame do mundo."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_quat_w


def instance_randomize_object_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a orientação do objeto no frame do mundo com randomização de instâncias."""
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 4), fill_value=-1)

    object: RigidObjectCollection = env.scene[object_cfg.name]
    object_quat_w = []
    for env_id in range(env.num_envs):
        object_quat_w.append(object.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
    return torch.stack(object_quat_w)


def object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Observações do objeto no frame do mundo:
        - Posição do objeto
        - Orientação do objeto
        - Vetor do gripper para o objeto
        - Altura do objeto em relação ao chão
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    object_quat_w = object.data.root_quat_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object = object_pos_w - ee_pos_w
    object_height = object_pos_w[:, 2:3]

    return torch.cat(
        (
            object_pos_w - env.scene.env_origins,
            object_quat_w,
            gripper_to_object,
            object_height,
        ),
        dim=1,
    )


def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """
    Observações do objeto no frame do mundo com randomização de instâncias:
        - Posição do objeto
        - Orientação do objeto
        - Vetor do gripper para o objeto
        - Altura do objeto em relação ao chão
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 9), fill_value=-1)

    object: RigidObjectCollection = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = []
    object_quat_w = []
    for env_id in range(env.num_envs):
        object_pos_w.append(object.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
        object_quat_w.append(object.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :4])
    object_pos_w = torch.stack(object_pos_w)
    object_quat_w = torch.stack(object_quat_w)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    gripper_to_object = object_pos_w - ee_pos_w
    object_height = object_pos_w[:, 2:3]

    return torch.cat(
        (
            object_pos_w - env.scene.env_origins,
            object_quat_w,
            gripper_to_object,
            object_height,
        ),
        dim=1,
    )


# ==========================
# End-Effector Observations
# ==========================

def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Retorna a posição do end-effector no frame do mundo."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :]


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """Retorna a orientação (quaternion) do end-effector no frame do mundo."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


# ==========================
# Gripper Observations
# ==========================

def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Retorna a posição do gripper (dedos) do robô."""
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)
    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


# ==========================
# Task-Specific Observations
# ==========================

def object_grasped(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_distance: float = 0.02, 
) -> torch.Tensor:
    """Verifica se o objeto foi agarrado pelo end-effector."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    object_pos = object.data.root_pos_w[:, :3]
    
    distance = torch.norm(ee_pos - object_pos, dim=1)
    return distance < grasp_distance


def object_stacked(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
) -> torch.Tensor:
    """Verifica se o objeto foi colocado no ponto final."""
    object: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    pos_diff = object.data.root_pos_w - goal.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    height_dist = torch.abs(pos_diff[:, 2])

    return torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold)



