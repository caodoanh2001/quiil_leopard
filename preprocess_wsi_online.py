from configs import configs
import os

# AdvMIL uses two levels (magnifications)
patch_level_2 = True
patch_level_1 = True

feats = True

# All commands
command_extract_patch_lv2 = " ".join([
    "python3 /workspace/CLAM/create_patches_fp.py",
    "--source", configs["online"]["source_path"],
    "--save_dir", configs["online"]["patch_path"],
    "--patch_size", str(configs["online"]["patch_size"]),
    "--patch_level", str(configs["online"]["patch_level"]),
    "--seg --patch"
])

command_extract_patch_lv1 = " ".join([
    "python3 /workspace/AdvMIL/tools/big_to_small_patching.py",
    configs["online"]["patch_path"], configs["online"]["patch_path_advmil"]
])

command_extract_features = " ".join([
    "CUDA_VISIBLE_DEVICES=0",
    "python3 /workspace/CLAM/extract_features_fp.py",
    "--data_h5_dir", configs["online"]["patch_path_advmil"],
    "--data_slide_dir", configs["online"]["source_path"],
    "--csv_path", os.path.join(configs["online"]["patch_path_advmil"], 'process_list_autogen.csv'),
    "--feat_dir", configs["online"]["feat_path"],
    "--features_type", "ctranspath",
    "--batch_size 512",
    "--slide_ext .tif"
])

if patch_level_2:
    # print("Extracting patches level 2 ...")
    os.system(command_extract_patch_lv2)

if patch_level_1:
    # print("Extracting patches level 1 ...")
    os.system(command_extract_patch_lv1)

if feats:
    # print("Extracting features ...")
    os.system(command_extract_features)