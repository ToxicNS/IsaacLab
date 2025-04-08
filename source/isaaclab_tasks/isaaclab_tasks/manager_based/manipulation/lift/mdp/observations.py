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


# def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Retorna a última ação executada pelo robô."""
#     if hasattr(env, "last_action"):
#         return env.last_action
#     raise AttributeError("O ambiente não possui o atributo 'last_action'.")


def joint_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Retorna as posições relativas das juntas do robô."""
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_pos


def joint_vel_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Retorna as velocidades relativas das juntas do robô."""
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_vel


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a posição do objeto no frame raiz do robô."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def cube_positions_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a posição do cubo no frame do mundo."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return torch.cat((cube.data.root_pos_w), dim=1)

def cube_orientations_in_world_frame(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Retorna a orientação do cubo no frame do mundo."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_state_w[:, 3:7]


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Retorna os comandos gerados, como a posição alvo do objeto."""
    if hasattr(env, "commands") and command_name in env.commands:
        return env.commands[command_name]
    # Retorna um tensor padrão se o atributo 'commands' não existir ou o comando não for encontrado
    return torch.zeros((1, 3), device="cpu")


def ee_frame_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_pos = ee_frame.data.target_pos_w[:, 0, :] - env.scene.env_origins[:, 0:3]

    return ee_frame_pos


def ee_frame_quat(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_frame_quat = ee_frame.data.target_quat_w[:, 0, :]

    return ee_frame_quat


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    finger_joint_1 = robot.data.joint_pos[:, -1].clone().unsqueeze(1)
    finger_joint_2 = -1 * robot.data.joint_pos[:, -2].clone().unsqueeze(1)

    return torch.cat((finger_joint_1, finger_joint_2), dim=1)


def reaching_object(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Verifica se o end-effector está próximo do objeto."""
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, :3]
    object_pos = object.data.root_pos_w[:, :3]

    distance = torch.norm(ee_pos - object_pos, dim=1)
    return distance < 0.05  # Retorna True se a distância for menor que 5 cm


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """
    Verifica se o objeto foi agarrado por múltiplos end-effectors em múltiplos ambientes.

    Args:
        env: O ambiente gerenciado.
        robot_cfg: Configuração do robô (opcional).
        ee_frame_cfg: Configuração do end-effector.
        object_cfg: Configuração do objeto.
        diff_threshold: Distância máxima para considerar que o objeto foi agarrado.
        gripper_open_val: Valor de abertura do gripper.
        gripper_threshold: Limite para considerar o gripper fechado.

    Returns:
        Um tensor booleano indicando se o objeto foi agarrado em cada ambiente.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
    )

    return grasped


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Verifica se o objeto foi levantado acima de uma altura mínima."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] > minimal_height