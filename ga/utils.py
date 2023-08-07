import os

def get_next_ga_dir(directory_path):
    num_digits = 3  # Number of digits for subdirectory names
    subdirectory_prefix = "ga"
    subdirectory_pattern = subdirectory_prefix + "{:0" + str(num_digits) + "}"

    existing_subdirs = [d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d)) and d.startswith(subdirectory_prefix)]

    if not existing_subdirs:
        # If no subdirectories exist, return "ga000"
        next_subdir_name = subdirectory_pattern.format(0)
    else:
        # Find the latest subdirectory name and extract the number part
        latest_subdir_name = max(existing_subdirs)
        latest_subdir_number = int(latest_subdir_name[2:])

        # Increment the number and format it with the specified number of digits
        next_subdir_number = latest_subdir_number + 1
        next_subdir_name = subdirectory_pattern.format(next_subdir_number)

    # Join the next subdirectory name with the input directory path
    next_subdirectory_path = os.path.join(directory_path, next_subdir_name)

    return next_subdirectory_path
