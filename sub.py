import h5py
import os

# Expande o caminho para o diretório inicial
input_file = os.path.expanduser("~/IsaacLab/datasets/lift_annotated/annotated_dataset_2demos.hdf5")

with h5py.File(input_file, "r") as f:
    def print_hdf5_structure(name, obj):
        print(f"{name}: {type(obj)}")
    f.visititems(print_hdf5_structure)