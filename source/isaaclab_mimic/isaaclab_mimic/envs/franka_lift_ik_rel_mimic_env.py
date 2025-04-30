# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class FrankaCubeLiftIKRelMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper class for Franka Cube Lift IK Rel env.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))


    def target_eef_pose_to_action(
        self, target_eef_pose_dict: dict, gripper_action_dict: dict, noise: float | None = None, env_id: int = 0
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            noise: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if noise is not None:
            noise = noise * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)


    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to axis angle form
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        # Handle invalid division for the case when delta_rotation_angle is close to zero
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}


    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}
    

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets a dictionary of termination signal flags for each subtask in a task. The flag is 1
        when the subtask has been completed and 0 otherwise.

        Args:
            env_ids: Environment indices to get the termination signals for. If None, all envs are considered.

        Returns:
            A dictionary termination signal flags (False or True) for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = dict()
        subtask_terms = self.obs_buf["subtask_terms"]
        
        # Approach subtask - true when robot is close enough to the object
        signals["approach_obj"] = subtask_terms["approach_obj"][env_ids]
        
        # Grasp subtask - true when object is grasped
        signals["grasp_obj"] = subtask_terms["grasp_obj"][env_ids]
        
        # Lift subtask - true when object is lifted above threshold
        signals["lift_obj"] = subtask_terms["lift_obj"][env_ids]
        
        return signals


## Adicionado NS
def get_demo_length(dataset_path: str, demo_index: int = 0) -> int:
    """
    Obtém o comprimento total de um demo a partir do arquivo HDF5 exportado.

    Args:
        dataset_path: Caminho para o arquivo HDF5 do dataset.
        demo_index: Índice do demo no dataset. Padrão é 0.

    Returns:
        O número total de frames no demo.
    """
    import h5py

    with h5py.File(dataset_path, "r") as f:
        demo_key = f"demo_{demo_index}"
        num_frames = len(f[f"{demo_key}/actions"])
    return num_frames

# Modificar a função ajustar_subtarefas para remover target_object_position
def ajustar_subtarefas(subtask_indices: dict, num_frames: int, min_offset: int = 5) -> dict:
    """
    Ajusta os índices das subtarefas para evitar sobreposição.
    """
    # Definir ordem explícita das subtarefas - remover target_object_position
    ordem_subtarefas = ["approach_obj", "grasp_obj", "lift_obj"]
    
    # Inicializar dicionário ordenado
    from collections import OrderedDict
    ajustados = OrderedDict()
    
    # Iniciar com o primeiro offset
    ultimo_fim = 0
    
    # Processar subtarefas na ordem correta
    for subtask in ordem_subtarefas:
        if subtask in subtask_indices:
            inicio, fim = subtask_indices[subtask]
            # Garantir que o início seja após o fim da subtarefa anterior
            inicio = max(inicio, ultimo_fim + min_offset)
            # Garantir que o fim seja maior que o início
            fim = min(max(fim, inicio + 1), num_frames)
            ajustados[subtask] = (inicio, fim)
            ultimo_fim = fim
    
    return ajustados

def validar_subtarefas(subtask_indices: dict) -> bool:
    """
    Valida se as subtarefas não se sobrepõem e estão em ordem cronológica.

    Args:
        subtask_indices: Dicionário com os índices das subtarefas.

    Returns:
        True se as subtarefas forem válidas, False caso contrário.
    """
    ultimo_fim = 0
    for subtask, (inicio, fim) in sorted(subtask_indices.items(), key=lambda x: x[1][0]):
        if inicio < ultimo_fim:
            print(f"Erro: Subtarefa {subtask} começa antes do fim da anterior.")
            return False
        ultimo_fim = fim
    return True

def calculate_dynamic_offsets(subtask_configs, min_offset=10, max_offset=30):
    """
    Calcula os offsets dinâmicos para as subtarefas com base no comprimento do demo.
    
    Args:
        subtask_configs: Lista de configurações das subtarefas.
        min_offset: Offset mínimo entre subtarefas.
        max_offset: Offset máximo entre subtarefas.
        
    Returns:
        Configurações das subtarefas com offsets ajustados.
    """
    adjusted_configs = []
    last_end_index = 0

    for subtask in subtask_configs[:-1]:  # Exclui a última subtarefa fictícia
        # Ajustar o offset inicial e final com base no último índice
        start_offset = max(last_end_index + min_offset, subtask.subtask_term_offset_range[0])
        end_offset = start_offset + (subtask.subtask_term_offset_range[1] - subtask.subtask_term_offset_range[0])

        # Garantir que o offset final não ultrapasse o máximo permitido
        end_offset = min(end_offset, last_end_index + max_offset)

        # Validar se o índice final respeita os limites
        if end_offset < start_offset:
            raise ValueError(
                f"Erro ao ajustar offsets: índice final ({end_offset}) menor que índice inicial ({start_offset})."
            )

        # Atualizar a configuração da subtarefa
        subtask.subtask_term_offset_range = (start_offset, end_offset)
        adjusted_configs.append(subtask)
        last_end_index = end_offset

    # Adicionar a última subtarefa fictícia sem alterações
    adjusted_configs.append(subtask_configs[-1])

    return adjusted_configs

    

def validate_subtask_order(subtask_configs):
    """
    Valida se as configurações das subtarefas estão na ordem correta
    para a tarefa de levantamento (lift).
    
    Args:
        subtask_configs: Lista de configurações das subtarefas.
        
    Returns:
        True se as subtarefas estiverem na ordem correta.
        
    Raises:
        ValueError: Se a ordem das subtarefas não corresponder ao fluxo esperado.
    """
    # Extrair os nomes das subtarefas e seus índices de início
    subtask_info = []
    for subtask in subtask_configs:
        name = subtask.subtask_term_signal
        # Assumir que o início da subtarefa é o primeiro valor no range de offset
        start_index = subtask.subtask_term_offset_range[0]
        subtask_info.append((name, start_index))
    
    # Verificar se temos alguma subtarefa
    if not subtask_info:
        raise ValueError("Nenhuma subtarefa encontrada!")
    
    # Ordenar subtarefas por índice de início
    sorted_subtasks = sorted(subtask_info, key=lambda x: x[1])
    
    # Se alguma subtarefa começa no frame 0, emitir um aviso
    for name, start_idx in sorted_subtasks:
        if start_idx == 0:
            print(f"Aviso: Subtarefa '{name}' começa no frame inicial (0).")
    
    # Verificar se alguma subtarefa tem índice de início negativo
    for name, start_idx in sorted_subtasks:
        if start_idx < 0:
            raise ValueError(f"Subtarefa '{name}' tem índice de início negativo: {start_idx}")
    
    # Verificar se as subtarefas estão em ordem cronológica
    for i in range(len(sorted_subtasks) - 1):
        curr_name, curr_idx = sorted_subtasks[i]
        next_name, next_idx = sorted_subtasks[i + 1]
        
        if curr_idx >= next_idx:
            raise ValueError(
                f"Ordem incorreta: '{curr_name}' (início: {curr_idx}) começa depois ou no mesmo frame que '{next_name}' (início: {next_idx})"
            )
    
    # Resultado final: subtarefas verificadas e em ordem cronológica
    return True
