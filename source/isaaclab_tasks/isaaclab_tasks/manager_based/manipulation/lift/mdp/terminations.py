# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.managers import TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def calculate_distance(ee_frame_cfg, object_cfg):
    """
    Calcula a distância entre o end-effector e o objeto.

    Args:
        ee_frame_cfg: Configuração do frame do end-effector.
        object_cfg: Configuração do objeto.

    Returns:
        A distância entre o end-effector e o objeto.
    """
    ee_pos = ee_frame_cfg.position  # Supondo que `position` contenha as coordenadas do end-effector
    object_pos = object_cfg.position  # Supondo que `position` contenha as coordenadas do objeto

    # Calcular a distância euclidiana
    distance = torch.norm(torch.tensor(ee_pos) - torch.tensor(object_pos))
    return distance 

def reaching_object(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Calcula a distância entre o end-effector e o objeto e retorna True se estiver dentro de um limite.

    Args:
        env: O ambiente.
        ee_frame_cfg: Configuração do frame do end-effector.
        object_cfg: Configuração do objeto.

    Returns:
        Um tensor indicando se o end-effector alcançou o objeto.
    """
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Posição do end-effector e do objeto
    ee_pos = ee_frame.data.target_pos_w[:, :3]  # Corrigido para usar target_pos_w
    object_pos = object.data.root_pos_w[:, :3]

    # Calcular a distância
    distance = torch.norm(ee_pos - object_pos, dim=1)

    # Retornar True se a distância for menor que um limite (exemplo: 0.05)
    return distance < 0.05



def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
    # Extrair as posições do robô e do objeto
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Calcular a posição desejada no frame do mundo
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)

    # Calcular a distância entre o objeto e a posição desejada
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # Retornar True se a distância for menor que o limite
    return distance < threshold


