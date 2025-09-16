"""
Script para treinamento distribuído em múltiplas GPUs utilizando PyTorch DistributedDataParallel (DDP).
Edite apenas as variáveis no topo do arquivo conforme necessário.
"""

# =============== CONFIGURAÇÕES DO USUÁRIO ===============
USE_ALL_GPUS = True
GPUS_TO_USE = [0, 1, 2, 3]

CONFIG_PATH = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/robomimic/bc_rnn_image_84.json"
DATASET_PATH = "./datasets/visuo/generated_dataset_failed.hdf5"
LOG_DIR = "robomimic_multi_gpu"
FORCE_OVERWRITE = True

OVERRIDE_CONFIG = {
    "train.batch_size": 128,
    # "train.num_epochs": 300,
}
MONITOR_GPU = True
NORMALIZE_ACTIONS = False
# ====================================================

import os
import sys
import time
import json
import torch
import traceback
import numpy as np
import h5py
import subprocess
import shutil
from threading import Thread
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import psutil
import torch.distributed as dist

# Isaac Sim
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Robomimic imports
import gymnasium as gym
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.algo import algo_factory
from robomimic.config import Config, config_factory
from robomimic.utils.log_utils import DataLogger, PrintLogger

# Isaac Lab imports (needed so that environment is registered)
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

def monitor_gpu_usage(rank):
    """Monitora o uso de GPU em uma thread separada."""
    if rank != 0:
        return
    while True:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.free', '--format=csv,noheader'],
                                stdout=subprocess.PIPE, text=True)
            print("\n=== GPU Usage ===")
            print(result.stdout)
            time.sleep(10)
        except Exception as e:
            print(f"Erro ao monitorar GPUs: {e}")
            break

def setup_distributed():
    """Inicializa o ambiente distribuído DDP."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        return 0, 1

def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    base, ext = os.path.splitext(config.train.data)
    normalized_path = base + "_normalized" + ext
    print(f"Criando dataset normalizado em {normalized_path}")
    shutil.copyfile(config.train.data, normalized_path)
    with h5py.File(normalized_path, "r+") as f:
        dataset_paths = [f"/data/demo_{str(i)}/actions" for i in range(len(f["data"].keys()))]
        dataset = np.array(f[dataset_paths[0]]).flatten()
        for i, path in enumerate(dataset_paths):
            if i != 0:
                data = np.array(f[path]).flatten()
                dataset = np.append(dataset, data)
        max_val = np.max(dataset)
        min_val = np.min(dataset)
        for i, path in enumerate(dataset_paths):
            data = np.array(f[path])
            normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
            del f[path]
            f[path] = normalized_data
        with open(os.path.join(log_dir, "normalization_params.txt"), "w") as ftxt:
            ftxt.write(f"min: {min_val}\n")
            ftxt.write(f"max: {max_val}\n")
    return normalized_path

def train(config: Config, device: torch.device, log_dir: str, ckpt_dir: str, video_dir: str, rank: int, world_size: int):
    torch.cuda.empty_cache()
    np.random.seed(config.train.seed + rank)
    torch.manual_seed(config.train.seed + rank)

    if rank == 0:
        print("\n============= New Training Run with Config =============")
        print(config)
        print("")
        print(f">>> Saving logs into directory: {log_dir}")
        print(f">>> Saving checkpoints into directory: {ckpt_dir}")
        print(f">>> Saving videos into directory: {video_dir}")

    if config.experiment.logging.terminal_output_to_txt and rank == 0:
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    ObsUtils.initialize_obs_utils_with_config(config)
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset at provided path {dataset_path} not found!")

    if rank == 0:
        print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data, all_obs_keys=config.all_obs_keys, verbose=(rank == 0)
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        if rank == 0:
            print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    envs = OrderedDict()
    if config.experiment.rollout.enabled and rank == 0:
        env_names = [env_meta["env_name"]]
        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)
        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name,
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
            )
            envs[env.name] = env
            print(envs[env.name])
    if rank == 0:
        print("")

    data_logger = DataLogger(log_dir, config=config, log_tb=config.experiment.logging.log_tb) if rank == 0 else None
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[rank])
    else:
        model = model.to(device)

    if rank == 0:
        with open(os.path.join(log_dir, "..", "config.json"), "w") as outfile:
            json.dump(config, outfile, indent=4)
        print("\n============= Model Summary =============")
        print(model)
        print("")

    trainset, validset = TrainUtils.load_data_for_training(config, obs_keys=shape_meta["all_obs_keys"])
    if world_size > 1:
        train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = trainset.get_dataset_sampler()
    if rank == 0:
        print("\n============= Training Dataset =============")
        print(trainset)
        print("")

    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=1,
        drop_last=True,
        pin_memory=True,
        persistent_workers=False
    )

    if config.experiment.validate:
        num_workers = min(config.train.num_data_workers, 1)
        if world_size > 1:
            valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True,
        )
    else:
        valid_loader = None

    best_valid_loss = None
    last_ckpt_time = time.time()
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps

    for epoch in range(1, config.train.num_epochs + 1):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
            if valid_loader is not None and hasattr(valid_loader.sampler, "set_epoch"):
                valid_loader.sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n======= INICIANDO EPOCH {epoch} ======")
            print(f"Simulador ativo: {simulation_app is not None}")
        step_log = TrainUtils.run_epoch(model=model, data_loader=train_loader, epoch=epoch, num_steps=train_num_steps)
        if hasattr(model, "on_epoch_end"):
            model.on_epoch_end(epoch)

        epoch_ckpt_name = f"model_epoch_{epoch}"
        should_save_ckpt = False
        if config.experiment.save.enabled and rank == 0:
            time_check = (config.experiment.save.every_n_seconds is not None) and (
                time.time() - last_ckpt_time > config.experiment.save.every_n_seconds
            )
            epoch_check = (
                (config.experiment.save.every_n_epochs is not None)
                and (epoch > 0)
                and (epoch % config.experiment.save.every_n_epochs == 0)
            )
            epoch_list_check = epoch in config.experiment.save.epochs
            should_save_ckpt = time_check or epoch_check or epoch_list_check
        ckpt_reason = None
        if should_save_ckpt and rank == 0:
            last_ckpt_time = time.time()
            ckpt_reason = "time"

        if rank == 0:
            print(f"Train Epoch {epoch}")
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record(f"Timing_Stats/Train_{k[5:]}", v, epoch)
                else:
                    data_logger.record(f"Train/{k}", v, epoch)

        if config.experiment.validate:
            with torch.no_grad():
                step_log = TrainUtils.run_epoch(
                    model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps
                )
            if rank == 0:
                for k, v in step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record(f"Timing_Stats/Valid_{k[5:]}", v, epoch)
                    else:
                        data_logger.record(f"Valid/{k}", v, epoch)
                print(f"Validation Epoch {epoch}")
                print(json.dumps(step_log, sort_keys=True, indent=4))
                valid_check = "Loss" in step_log
                if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                    best_valid_loss = step_log["Loss"]
                    if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                        epoch_ckpt_name += f"_best_validation_{best_valid_loss}"
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

        if should_save_ckpt and rank == 0:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
            )

        if rank == 0:
            process = psutil.Process(os.getpid())
            mem_usage = int(process.memory_info().rss / 1000000)
            data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
            print(f"\nEpoch {epoch} Memory Usage: {mem_usage} MB\n")
            if torch.cuda.device_count() > 1:
                for i in range(torch.cuda.device_count()):
                    gpu_mem = torch.cuda.memory_allocated(i)/1024/1024
                    print(f"GPU {i} memória: {gpu_mem:.2f} MB")
                    data_logger.record(f"System/GPU_{i}_Memory_MB", gpu_mem, epoch)
            if epoch % 5 == 0:
                torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    if rank == 0:
        data_logger.close()

def main():
    rank, world_size = setup_distributed()

    if USE_ALL_GPUS:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            if rank == 0:
                print("AVISO: Nenhuma GPU detectada. Utilizando CPU.")
            GPUS_TO_USE = []
        else:
            GPUS_TO_USE = list(range(num_gpus))
            if rank == 0:
                print(f"Detectadas {num_gpus} GPUs: {GPUS_TO_USE}")
                for i in GPUS_TO_USE:
                    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    print(f"  Memória total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    if rank == 0:
        print(f"Configurando GPUs: {','.join(map(str, GPUS_TO_USE))}")

    torch.cuda.set_device(rank)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if MONITOR_GPU and rank == 0:
        gpu_thread = Thread(target=monitor_gpu_usage, args=(rank,), daemon=True)
        gpu_thread.start()

    if rank == 0:
        print(f"GPUs disponíveis: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        if rank == 0:
            print(f"Carregando configuração do arquivo: {CONFIG_PATH}")
        with open(CONFIG_PATH) as f:
            ext_cfg = json.load(f)
            config = config_factory(ext_cfg["algo_name"])
        with config.values_unlocked():
            config.update(ext_cfg)
            config.train.data = DATASET_PATH
            for key, value in OVERRIDE_CONFIG.items():
                parts = key.split(".")
                curr = config
                for part in parts[:-1]:
                    if hasattr(curr, part):
                        curr = getattr(curr, part)
                    else:
                        if rank == 0:
                            print(f"Aviso: Caminho de configuração inválido: {key}")
                        break
                else:
                    setattr(curr, parts[-1], value)
                    if rank == 0:
                        print(f"Substituído {key} = {value}")
        config.train.output_dir = os.path.abspath(os.path.join("./logs", LOG_DIR))
        log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)
        if NORMALIZE_ACTIONS and rank == 0:
            config.train.data = normalize_hdf5_actions(config, log_dir)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        config.lock()
        global simulation_app
        try:
            train(config, device, log_dir, ckpt_dir, video_dir, rank, world_size)
            if rank == 0:
                print("Treinamento concluído com sucesso!")
        except Exception as e:
            if rank == 0:
                print(f"Erro ao inicializar treinamento: {e}\n{traceback.format_exc()}")
    except Exception as e:
        if rank == 0:
            print(f"Erro na configuração: {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()