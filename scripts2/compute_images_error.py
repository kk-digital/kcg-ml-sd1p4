import sys
import torch
import safetensors.torch as st
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

device = get_device()

if len(sys.argv) == 1:
    grid_init_pt_path = abspath("./output/grid_all_Logistic0.49_0.54.safetensors")
    grid_disk_pt_path = abspath("./output/grid_all_Logistic0.49_0.54_from_disk.safetensors")
    grid_init = st.load_file(grid_init_pt_path, device=device)
    grid_init_st = list(grid_init.values())[0]
    grid_disk = st.load_file(grid_init_pt_path, device=device)
    grid_disk_st = list(grid_disk.values())[0]
    print("Using safetensors")

grid_init_pt_path = abspath("./output/grid_all_Logistic0.49_0.54.pt")
grid_disk_pt_path = abspath("./output/grid_all_Logistic0.49_0.54_from_disk.pt")
grid_init = torch.load(grid_init_pt_path, map_location=device)
grid_disk = torch.load(grid_disk_pt_path, map_location=device)
print("Using torch tensors")

grids_diff = grid_init - grid_disk
print("diff, using torch tensors: ", torch.norm(grids_diff))

if len(sys.argv) == 1:
    assert torch.equal(grid_init_st, grid_init)
    assert torch.equal(grid_disk_st, grid_disk)
    
    grids_diff = grid_init_st - grid_disk_st
    print("diff, using safetensors: ", torch.norm(grids_diff))