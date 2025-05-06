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

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms

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
    ee_pos = ee_frame_cfg.position  
    object_pos = object_cfg.position 

    # Calcular a distância euclidiana
    distance = torch.norm(torch.tensor(ee_pos) - torch.tensor(object_pos))
    return distance 

# def object_reached_goal(
#     env: ManagerBasedRLEnv,
#     command_name: str = "object_pose",
#     threshold: float = 0.05,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Verifica se o objeto chegou à posição final (goal position).

#     Args:
#         env: O ambiente.
#         command_name: O nome do comando que define a posição alvo.
#         threshold: A distância máxima para considerar que o objeto atingiu a meta. Padrão é 0.05.
#         robot_cfg: A configuração do robô. Padrão é SceneEntityCfg("robot").
#         object_cfg: A configuração do objeto. Padrão é SceneEntityCfg("object").

#     Returns:
#         Tensor booleano indicando se o objeto atingiu a posição alvo.
#     """
#     # Obter os objetos necessários
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
    
#     # Calcular a posição alvo no frame do mundo
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    
#     # Calcular a distância entre a posição atual do objeto e a posição alvo
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    
#     # Retorna True se a distância for menor que o threshold
#     return distance < threshold

def object_reached_goal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.05,
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Verifica se o objeto chegou à posição final (goal position) no frame do robô.

    Args:
        env: O ambiente.
        threshold: A distância máxima para considerar que o objeto atingiu a meta. Padrão é 0.05.
        command_name: O nome do comando que define a posição alvo.
        robot_cfg: A configuração do robô. Padrão é SceneEntityCfg("robot").
        object_cfg: A configuração do objeto. Padrão é SceneEntityCfg("object").

    Returns:
        Tensor booleano indicando se o objeto atingiu a posição alvo.
    """
    # Obter os objetos necessários
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Obter a posição alvo do comando
    command = env.command_manager.get_command(command_name)
    goal_position = command[:, :3]

    # Posição do objeto no frame do robô
    object_pos_r, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object.data.root_pos_w[:, :3]
    )

    # Certificar-se de que goal_position está no mesmo dispositivo que object_pos_r
    goal_position = goal_position.to(object_pos_r.device)

    # Calcular a distância entre a posição atual do objeto e a posição alvo no frame do robô
    distance = torch.norm(object_pos_r - goal_position, dim=1)

    # Retorna True se a distância for menor que o threshold
    return distance < threshold