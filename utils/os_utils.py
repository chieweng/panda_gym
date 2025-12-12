import os
import shutil

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def backup_and_clean(directory):
    "Moves all existing files in `directory` to a subfolder `backup`"
    backup_folder = os.path.join(directory, "backup")

    # Remove old backups to prevent piling
    if os.path.exists(backup_folder):
        shutil.rmtree(backup_folder)

    os.makedirs(backup_folder, exist_ok=True)  # Initializes new backup folder

    # Move all files from the main directory to the backup folder
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  # Only move files, not folders
            shutil.move(file_path, os.path.join(backup_folder, filename))

    logging.info(f"All files moved to {backup_folder}.")