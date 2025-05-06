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
from isaaclab.utils.math import quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ==========================
# Robot State Observations
# ==========================

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
# ====== Position ======
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

def instance_randomize_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """
    Retorna a posição do objeto no frame raiz do robô com randomização de instâncias.
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 3), fill_value=-1)

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObjectCollection = env.scene[object_cfg.name]

    object_pos_w = []
    for env_id in range(env.num_envs):
        object_pos_w.append(object.data.object_pos_w[env_id, env.rigid_objects_in_focus[env_id][0], :3])
    object_pos_w = torch.stack(object_pos_w)

    object_pos_r, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_r

# ====== Orientation ======
def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    Retorna a orientação do objeto no frame raiz do robô.

    Args:
        env: O ambiente.
        robot_cfg: Configuração do robô.
        object_cfg: Configuração do objeto.

    Returns:
        torch.Tensor: Orientação do objeto no frame raiz do robô.
    """
    # robot: RigidObject = env.scene[robot_cfg.name]
    # object: RigidObject = env.scene[object_cfg.name]
    # object_quat_w = object.data.root_quat_w
    # object_ori_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_quat_w
    # )
    # return object_ori_b
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_quat_w = object.data.root_quat_w

    # Use quat_inv para calcular o inverso do quaternion do robô
    robot_quat_inv = quat_inv(robot.data.root_state_w[:, 3:7])

    # Multiplique o inverso do quaternion do robô pela orientação do objeto
    object_ori_b = quat_mul(robot_quat_inv, object_quat_w)

    return object_ori_b

def instance_randomize_object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """
    Retorna a orientação do objeto no frame raiz do robô, considerando instâncias específicas
    em ambientes paralelos. Se os objetos não estiverem em foco, retorna um tensor preenchido com -1.

    Args:
        env: O ambiente de simulação.
        robot_cfg: Configuração do robô.
        object_cfg: Configuração do objeto.

    Returns:
        torch.Tensor: Orientação do objeto no frame raiz do robô ou -1 se não estiver em foco.
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 4), fill_value=-1)  # Quaternion tem 4 componentes

    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObjectCollection = env.scene[object_cfg.name]

    object_quat_w = []
    for env_id in range(env.num_envs):
        # Verificar se há objetos em foco no ambiente atual
        if env.rigid_objects_in_focus[env_id]:
            object_quat_w.append(object.data.object_quat_w[env_id, env.rigid_objects_in_focus[env_id][0], :])
        else:
            object_quat_w.append(torch.full((4,), fill_value=-1))  # Preencher com -1 se não houver foco

    object_quat_w = torch.stack(object_quat_w)

    # Transformar a orientação do objeto para o frame raiz do robô
    object_ori_r, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_quat_w
    )
    return object_ori_r

# ==========================
# Object Observations
# ==========================
def object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Observações detalhadas do estado de um único objeto e do manipulador:
        - Posição do objeto relativa à base do robô
        - Orientação do objeto relativa à base do robô
        - Vetor do gripper para o objeto
        - Altura do objeto em relação ao chão

    Args:
        env: O ambiente de simulação.
        object_cfg: Configuração do objeto.
        ee_frame_cfg: Configuração do frame do end-effector.
        robot_cfg: Configuração do robô.

    Returns:
        torch.Tensor: Observações concatenadas.
    """
    # Usar funções auxiliares para calcular posição e orientação relativas à base do robô
    object_pos_relative_to_robot = object_position_in_robot_root_frame(
        env, robot_cfg=robot_cfg, object_cfg=object_cfg
    )
    object_orientation_relative_to_robot = object_orientation_in_robot_root_frame(
        env, robot_cfg=robot_cfg, object_cfg=object_cfg
    )

    # Calcular relações espaciais
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    object: RigidObject = env.scene[object_cfg.name]
    gripper_to_object = object.data.root_pos_w[:, :3] - ee_pos_w
    object_height = object.data.root_pos_w[:, 2:3]

    # Retornar as observações concatenadas
    return torch.cat(
        (
            object_pos_relative_to_robot,         # Posição relativa à base do robô
            object_orientation_relative_to_robot, # Orientação relativa à base do robô
            gripper_to_object,                    # Vetor do gripper para o objeto
            object_height,                        # Altura do objeto
        ),
        dim=1,
    )

def instance_randomize_object_obs(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Observações detalhadas do estado de um único objeto e do manipulador, considerando
    múltiplos ambientes paralelos e instâncias específicas.

    Args:
        env: O ambiente de simulação.
        object_cfg: Configuração do objeto.
        ee_frame_cfg: Configuração do frame do end-effector.
        robot_cfg: Configuração do robô.

    Returns:
        torch.Tensor: Observações concatenadas ou -1 se o objeto não estiver em foco.
    """
    if not hasattr(env, "rigid_objects_in_focus"):
        return torch.full((env.num_envs, 10), fill_value=-1)  # 10 é o número total de features concatenadas

    object_pos_relative_to_robot = []
    object_orientation_relative_to_robot = []
    gripper_to_object = []
    object_height = []

    for env_id in range(env.num_envs):
        if env.rigid_objects_in_focus[env_id]:
            # Usar funções auxiliares para calcular posição e orientação relativas à base do robô
            pos = object_position_in_robot_root_frame(
                env, robot_cfg=robot_cfg, object_cfg=object_cfg
            )[env_id]
            ori = object_orientation_in_robot_root_frame(
                env, robot_cfg=robot_cfg, object_cfg=object_cfg
            )[env_id]

            # Calcular relações espaciais
            ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[env_id, 0, :]
            obj_pos = env.scene[object_cfg.name].data.root_pos_w[env_id, :3]
            gripper_to_obj = obj_pos - ee_pos
            height = obj_pos[2:3]

            # Adicionar às listas
            object_pos_relative_to_robot.append(pos)
            object_orientation_relative_to_robot.append(ori)
            gripper_to_object.append(gripper_to_obj)
            object_height.append(height)
        else:
            # Preencher com -1 se o objeto não estiver em foco
            object_pos_relative_to_robot.append(torch.full((3,), fill_value=-1))
            object_orientation_relative_to_robot.append(torch.full((4,), fill_value=-1))  # Quaternion
            gripper_to_object.append(torch.full((3,), fill_value=-1))
            object_height.append(torch.full((1,), fill_value=-1))

    # Concatenar todas as observações
    return torch.cat(
        (
            torch.stack(object_pos_relative_to_robot),
            torch.stack(object_orientation_relative_to_robot),
            torch.stack(gripper_to_object),
            torch.stack(object_height),
        ),
        dim=1,
    )


# ==========================
# End-Effector Observations
# ==========================

def ee_frame_pos(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Retorna a posição do end-effector no frame raiz do robô."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_frame_pos, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_frame_pos


def ee_frame_quat(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Retorna a orientação do end-effector no frame raiz do robô."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # Orientação do end-effector (quaternion)
    robot_quat_w = robot.data.root_state_w[:, 3:7]  # Orientação do robô (quaternion)

    # Calcular a orientação relativa usando quaternions
    robot_quat_inv = quat_inv(robot_quat_w)  # Inverso do quaternion do robô
    ee_frame_quat = quat_mul(robot_quat_inv, ee_quat_w)  # Multiplicar pelo quaternion do end-effector

    return ee_frame_quat


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
    grasp_distance: float = 0.05,  # Distância máxima para considerar o objeto agarrado
    lift_threshold: float = 0.005,  # Altura mínima para considerar que o objeto foi levantado (0.5 cm)
) -> torch.Tensor:
    """Detecta o início e término da subtarefa 'grasp_obj'.

    A subtarefa começa quando o end-effector está a menos de `grasp_distance` do objeto
    e termina quando o objeto é levantado acima de `lift_threshold`.

    Args:
        env: O ambiente.
        ee_frame_cfg: Configuração do frame do end-effector.
        object_cfg: Configuração do objeto.
        grasp_distance: Distância máxima para considerar que o objeto foi agarrado.
        lift_threshold: Altura mínima para considerar que o objeto foi levantado.

    Returns:
        True enquanto o objeto está sendo agarrado e False quando o objeto é levantado.
    """
    # Obter o end-effector e o objeto
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    # Armazenar a altura inicial do objeto no início do episódio
    if not hasattr(env, "object_initial_height"):
        env.object_initial_height = object.data.root_pos_w[:, 2].clone()

    # Posições no espaço
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    object_pos = object.data.root_pos_w[:, :3]

    # Calcular a distância entre o end-effector e o objeto
    distance = torch.norm(ee_pos - object_pos, dim=1)
    object_near = distance < grasp_distance

    # Verificar se o objeto foi levantado acima do limite
    object_lifted = object.data.root_pos_w[:, 2] > (env.object_initial_height + lift_threshold)

    # A subtarefa termina quando o objeto é levantado
    return torch.logical_and(object_near, torch.logical_not(object_lifted))

# def object_grasped(
#     env: ManagerBasedRLEnv,
#     robot_cfg: SceneEntityCfg,
#     ee_frame_cfg: SceneEntityCfg,
#     object_cfg: SceneEntityCfg,
#     diff_threshold: float = 0.06,
#     gripper_open_val: torch.tensor = torch.tensor([0.04]),
#     gripper_threshold: float = 0.005,
# ) -> torch.Tensor:
#     """Check if an object is grasped by the specified robot."""

#     robot: Articulation = env.scene[robot_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]

#     object_pos = object.data.root_pos_w
#     end_effector_pos = ee_frame.data.target_pos_w[:, 0, :]
#     pose_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

#     grasped = torch.logical_and(
#         pose_diff < diff_threshold,
#         torch.abs(robot.data.joint_pos[:, -1] - gripper_open_val.to(env.device)) > gripper_threshold,
#     )
#     grasped = torch.logical_and(
#         grasped, torch.abs(robot.data.joint_pos[:, -2] - gripper_open_val.to(env.device)) > gripper_threshold
#     )

#     return grasped


def object_approached(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_velocity: float = 0.001,  # Movimento mínimo para detectar início
    threshold: float = 0.05,  # 5 cm para término
) -> torch.Tensor:
    """Subtarefa 'approach_obj': Detecta quando o end-effector se aproxima do objeto.
    
    Esta função retorna True quando o end-effector está a menos de 5cm do objeto.
    Representa o movimento inicial até chegar perto do objeto.
    
    Args:
        env: O ambiente.
        robot_cfg: Configuração do robô.
        object_cfg: Configuração do objeto.
        threshold: Distância máxima para considerar que se aproximou (5cm).
        
    Returns:
        True quando o end-effector está próximo do objeto.
    """
    # Obter o robô, end-effector e objeto
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # Verificar se o robô está em movimento (baseado na velocidade das juntas)
    joint_vel_magnitude = torch.norm(robot.data.joint_vel, dim=1)
    robot_moving = joint_vel_magnitude > min_velocity
    
    # Calcular a distância entre o end-effector e o objeto
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    object_pos = object.data.root_pos_w[:, :3]
    distance = torch.norm(ee_pos - object_pos, dim=1)
    ee_near_object = distance < threshold
    
    # A subtarefa está ativa enquanto o robô está se movendo e o end-effector não está próximo
    return torch.logical_and(robot_moving, torch.logical_not(ee_near_object))


def object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    lift_start: float = 0.005,  # Altura mínima para considerar que o objeto foi levantado (0.5 cm)
    lift_end: float = 0.10,  # Altura máxima para considerar que o objeto foi levantado (10 cm)
) -> torch.Tensor:
    """Detecta o início e término da subtarefa 'lift_obj'.

    A subtarefa começa quando o objeto é levantado acima de `lift_start`
    em relação à sua altura inicial e termina quando o objeto atinge `lift_end`.

    Args:
        env: O ambiente.
        object_cfg: Configuração do objeto.
        lift_start: Altura mínima para considerar que o objeto foi levantado.
        lift_end: Altura máxima para considerar que o objeto foi levantado.

    Returns:
        True enquanto o objeto está sendo levantado e False quando atinge a altura máxima.
    """
    # Obter o objeto
    object: RigidObject = env.scene[object_cfg.name]

    # Altura inicial do objeto (armazenada no início do episódio)
    if not hasattr(env, "object_initial_height"):
        env.object_initial_height = object.data.root_pos_w[:, 2].clone()

    # Verificar se o objeto está dentro do intervalo de levantamento
    object_height = object.data.root_pos_w[:, 2]
    object_lifted = object_height > (env.object_initial_height + lift_start)
    object_not_too_high = object_height < (env.object_initial_height + lift_end)

    # A subtarefa está ativa enquanto o objeto está dentro do intervalo
    return torch.logical_and(object_lifted, object_not_too_high)



