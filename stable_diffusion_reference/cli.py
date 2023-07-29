import os

def check_folder_existence(folder_path: str):
    """
    ### Check if a folder exists

    :param folder_path: is the path of the folder
    """
    os.makedirs(folder_path, exist_ok=True)

def check_file_existence(file_path: str):
    """
    ### Check if a file exists

    :param file_path: is the path of the file
    """
    if not os.path.isfile(file_path):
        print("ERROR: File '{}' does not exist".format(file_path))
        exit(1)