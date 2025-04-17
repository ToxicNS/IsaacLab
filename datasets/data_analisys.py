import h5py

# def get_all_keys_in_demo(h5file, demo_id="demo_1"):
#     keys = []
#     base_path = f"data/{demo_id}"
#     if base_path in h5file:
#         h5file[base_path].visititems(lambda name, obj: keys.append(f"{base_path}/{name}"))
#     return set(keys)

# def compare_demo_keys(file1_path, file2_path, demo_id="demo_1"):
#     with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
#         keys1 = get_all_keys_in_demo(f1, demo_id)
#         keys2 = get_all_keys_in_demo(f2, demo_id)

#         only_in_file1 = sorted(keys1 - keys2)
#         only_in_file2 = sorted(keys2 - keys1)
#         common_keys = sorted(keys1 & keys2)

#         print(f"\n=== Comparando chaves em 'data/{demo_id}' entre os arquivos ===")
#         print(f"\n🔑 Total de chaves em file1: {len(keys1)}")
#         print(f"🔑 Total de chaves em file2: {len(keys2)}")
#         print(f"🤝 Chaves em comum: {len(common_keys)}")

#         if only_in_file1:
#             print("\n❌ Chaves **apenas no file1**:")
#             for key in only_in_file1:
#                 print(f"  {key}")

#         if only_in_file2:
#             print("\n❌ Chaves **apenas no file2**:")
#             for key in only_in_file2:
#                 print(f"  {key}")

#         if common_keys:
#             print("\n✅ Chaves em comum:")
#             for key in common_keys:
#                 print(f"  {key}")

# # Caminhos dos ficheiros
# file1 = "/home/lab4/IsaacLab/datasets/lift_generated/generated_dataset_small_10r_10demos.hdf5"
# file2 = "/home/lab4/dataset_good/generated_dataset_small_2.hdf5"

# # Demo a comparar
# compare_demo_keys(file1, file2, demo_id="demo_1")


import h5py

def check_mask_keys(file_path):
    print(f"\n🔍 Verificando o conteúdo de {file_path}")
    with h5py.File(file_path, "r") as f:
        if "mask" in f:
            print("🟢 Chaves dentro de /mask:")
            for key in f["mask"].keys():
                print("  -", key)
        else:
            print("🔴 O grupo /mask não existe no ficheiro!")

# Substitui pelos teus caminhos:
check_mask_keys("/home/lab4/IsaacLab/datasets/lift_generated/generated_dataset_small_10r_10demos.hdf5")
check_mask_keys("/home/lab4/dataset_good/generated_dataset_small_2.hdf5")
