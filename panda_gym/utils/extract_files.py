import os
import re
import numpy as np

def extract_files(input_folder, output_root):
    """
    Dynamically extract .npy arrays from .npz files in either folder structure:
    1. collision_labels/scene_xxxx/*.npz
    2. grasp_labels/*.npz (each .npz is a scene)
    """
    parent_folder_name = os.path.basename(input_folder.rstrip("/"))
    output_parent_dir = os.path.join(output_root, parent_folder_name)
    os.makedirs(output_parent_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if not file.endswith(".npz"):
                continue

            npz_path = os.path.join(root, file)

            # Secure extraction of scene ID (leading digits)
            match = re.match(r"(\d+)", file)
            if match:
                scene_id = int(match.group(1))
            else:
                # fallback: use parent folder name
                match_folder = re.match(r".*?(\d+)", os.path.basename(root))
                if match_folder:
                    scene_id = int(match_folder.group(1))
                else:
                    raise ValueError(f"Cannot extract scene number from {file}")

            # Make output folder name, zero-padded to 4 digits
            scene_folder = f"scene_{scene_id:04d}"
            scene_output_dir = os.path.join(output_parent_dir, scene_folder)
            os.makedirs(scene_output_dir, exist_ok=True)

            extract_npz(npz_path, scene_output_dir)



def extract_npz(npz_file_path, scene_output_dir):
    """
    Extract arrays from a single .npz file.
    """

    with np.load(npz_file_path) as data:
        for array_name in data.files:
            arr = data[array_name]

            out_filename = f"{array_name}.npy"
            out_path = os.path.join(scene_output_dir, out_filename)

            np.save(out_path, arr)


collision_label = "../graspnet/collision_label"
grasp_label = "../graspnet/grasp_label"
output_folder = "../graspnet"

os.makedirs(output_folder, exist_ok=True)

# extract_files(collision_label, output_folder)
extract_files(grasp_label, output_folder)
