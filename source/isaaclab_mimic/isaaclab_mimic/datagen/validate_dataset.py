import h5py

def inspect_hdf5(file_path):
    """
    Inspeciona a estrutura de um arquivo HDF5.

    Args:
        file_path (str): Caminho para o arquivo HDF5.
    """
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            print(name)
        f.visititems(print_structure)

# Exemplo de uso
inspect_hdf5('./datasets/lift_annotated/annotated_dataset_2demos.hdf5')


# import h5py

# def validate_dataset(file_path):
#     """
#     Valida o dataset para garantir que os índices das subtarefas estão corretos.

#     Args:
#         file_path (str): Caminho para o arquivo HDF5 do dataset.
#     """
#     with h5py.File(file_path, 'r') as f:
#         for episode_name in f.keys():
#             episode = f[episode_name]
#             actions = episode['actions'][:]
#             subtask_signals = episode['obs']['subtask_term_signals'][:]

#             # Verificar se os índices das subtarefas estão ordenados
#             prev_end_index = 0
#             for i, signal in enumerate(subtask_signals):
#                 end_index = (signal[1:] - signal[:-1]).nonzero()[0][0] + 1 if signal.any() else len(actions)
#                 assert prev_end_index < end_index, (
#                     f"Erro no episódio {episode_name}: Subtarefa {i} termina em {prev_end_index} "
#                     f"mas a próxima subtarefa começa em {end_index}."
#                 )
#                 prev_end_index = end_index

#     print("Dataset validado com sucesso!")

# # Exemplo de uso
# validate_dataset('./datasets/lift_annotated/annotated_dataset_2demos.hdf5')