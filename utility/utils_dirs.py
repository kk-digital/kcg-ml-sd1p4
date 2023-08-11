import os
import shutil

def create_folder_if_not_exist(folder_path):
    """
    Create a folder if it does not exist.

    Parameters:
        folder_path (str): The path of the folder to create.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def remove_all_files_and_folders(path):
    """
    Remove all files and folders inside the specified path.

    Parameters:
        path (str): The path for which to remove all files and folders.

    Returns:
        None
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)

    print(f"All files and folders inside '{path}' have been removed.")