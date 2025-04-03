# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b



from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation


def reaching_object(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    print(f"ee_frame_cfg: {ee_frame_cfg}, object_cfg: {object_cfg}")
    """Verifica se o end-effector está próximo do objeto."""
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, :3]
    object_pos = object.data.root_pos_w[:, :3]

    distance = torch.norm(ee_pos - object_pos, dim=1)
    return distance < 0.05  # Retorna True se a distância for menor que 5 cm


def object_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = None,  # Torna opcional
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    diff_threshold: float = 0.06,
    gripper_open_val: torch.tensor = torch.tensor([0.04]),
    gripper_threshold: float = 0.005,
) -> torch.Tensor:
    """Verifica se o objeto foi agarrado."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(object_pos - ee_pos, dim=1)

    # Verifica se robot_cfg foi fornecido
    if robot_cfg is not None:
        robot: Articulation = env.scene[robot_cfg.name]
        gripper_state = torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device))
        grasped = torch.logical_and(pose_diff < diff_threshold, gripper_state > gripper_threshold)
    else:
        grasped = pose_diff < diff_threshold  # Apenas verifica a distância

    return grasped


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Verifica se o objeto foi levantado acima de uma altura mínima."""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_pos_w[:, 2] > minimal_height