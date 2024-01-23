import os
import shutil
import numpy as np
from utils import create_safe_directory

# With this script we can get multiple subsets. For that, select them using the 
# variable below, and the following codes:
#   1: mini-serrano

mode = 1

def meet_condition(value, mode):
    if mode == 1:
        return ("sphere" in value or "blob" in value) and ("studio" in value or "tiber" in value or "cathedral" in value)
    else:
        print(f"[ERROR] Invalid mode: {mode}")
        os.exit()

# Source folder
folder_path = "color_chnl"

# Destiny folder
if(mode == 1):
    train_path = "mini-serrano"

create_safe_directory(train_path)

# Listing all files
files = os.listdir(folder_path)

save_idx = False


if save_idx: # ? To save the indices of the images
    print('Copying indices...')
    indices = [idx for idx,value in enumerate(files) if meet_condition(value,mode)]
    np.save(f"{train_path}_idx", indices)
else: # ? To save the actual images
    print('Copying files...')
    for file in files:
        # if "vrip" in file and "art_studio" in file:
        #     shutil.copy(os.path.join(folder_path, file), train_path)
        #     #a += 1
        # if "teapot" in file and "tiber" in file:
        #     shutil.copy(os.path.join(folder_path, file), train_path)
            #b += 1
        if meet_condition(file,mode):
            shutil.copy(os.path.join(folder_path, file), train_path)
            #c += 1
    #print(f'Got {a}, {b} and {c} samples')
