# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import h5py


from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.lift_ik_rel_blueprint_env_cfg import (
    FrankaCubeLiftBlueprintEnvCfg,
)
from isaaclab_mimic.envs.franka_lift_ik_rel_mimic_env import (
    ajustar_subtarefas,
    get_demo_length,  
    validar_subtarefas,
    calculate_dynamic_offsets,
    validate_subtask_order,
)

@configclass
class FrankaCubeLiftIKRelBlueprintMimicEnvCfg(FrankaCubeLiftBlueprintEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config class for Franka Lift Cube IK Rel env.
    """

    def __post_init__(self):
        # post init of parents
        super().__post_init__()
        
        # Verificar se o dataset_path está configurado
        if not hasattr(self.datagen_config, "dataset_path") or not self.datagen_config.dataset_path:
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--input_file", type=str, required=True, help="Caminho para o dataset")
            args, _ = parser.parse_known_args()
            self.datagen_config.dataset_path = args.input_file


        # Override the existing values
        self.datagen_config.name = "isaac_lab_franka_lift_ik_rel_blueprint_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        # The following are the subtask configurations for the lift task.
        subtask_configs = []

        # Aproximação do objeto
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

        # Agarrar o objeto
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

        # Levantar o objeto
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

        # Levar o objeto para a posição alvo
        subtask_configs.append(
            SubTaskConfig(
                object_ref="object",
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),  # Ajuste para evitar inconsistências
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.03,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        # 3. Ajustar offsets dinamicamente, mas SOMENTE para as subtarefas não-finais
        temp_configs = calculate_dynamic_offsets(subtask_configs, min_offset=10, max_offset=30)

        # 4. Adicionar a última subtarefa sem modificar seu offset
        self.subtask_configs["franka"] = temp_configs

        # Validar a ordem das subtarefas
        validate_subtask_order(self.subtask_configs["franka"])

        # Buscar o tamanho dos demos e ajustar as subtarefas
        import h5py
        with h5py.File(self.datagen_config.dataset_path, "r") as f:
            demo_keys = [key for key in f.keys() if key.startswith("demo_")]
            for demo_key in demo_keys:
                demo_index = int(demo_key.split("_")[1])  # Extrair o índice do demo
                num_frames = get_demo_length(self.datagen_config.dataset_path, demo_index)

                # Ajustar subtarefas para o demo atual
                self.subtask_configs["franka"] = ajustar_subtarefas(self.subtask_configs["franka"], num_frames)

                # Validar subtarefas ajustadas
                if not validar_subtarefas(self.subtask_configs["franka"]):
                    raise ValueError(f"Subtarefas ajustadas são inválidas para demo {demo_index}!")