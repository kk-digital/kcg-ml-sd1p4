import torch
from os.path import join, abspath

def get_device(force_cpu: bool = False, cuda_fallback: str = 'cuda:0'):
    """
    ### Get device
    """
    if torch.cuda.is_available() and not force_cpu:
        device_name = torch.cuda.get_device_name(0)
        print("INFO: Using CUDA device: {}".format(device_name))
        return cuda_fallback

    print("WARNING: You are running this script without CUDA. Brace yourself for a slow ride.")
    return 'cpu'

grid_init_pt_path = abspath("./output/grid_all_Logistic0.49_0.54.pt")
grid_disk_pt_path = abspath("./output/grid_all_Logistic0.49_0.54_from_disk.pt")
device = get_device()

grid_init = torch.load(grid_init_pt_path, map_location=device)
grid_disk = torch.load(grid_disk_pt_path, map_location=device)
grids_diff = grid_init - grid_disk


print(torch.norm(grids_diff))
