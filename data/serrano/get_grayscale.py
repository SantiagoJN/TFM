import os
import cv2
import shutil
import numpy as np
from utils import create_safe_directory

# Following the same strategy as before, we can build different datasets moduling
# the mode variable:
#   1: gray-spheres
#   2: gray-blobs
#   3: gray-buddhas
#   4: gray-dragons

mode = 4
if mode == 1:
    geometry = "spheres"
elif mode == 2:
    geometry = "blobs"
elif mode == 3:
    geometry = "buddhas"
elif mode == 4:
    geometry = "dragons"
else:
    print(f"[ERROR] Invalid mode: {mode}")
    exit()

# Source folder
folder_path = f"masked-{geometry}"

# Destiny folder
train_path = f"gray-{geometry}"
create_safe_directory(train_path)

# Listing all files
files = os.listdir(folder_path)

print('Copying files...')
for file in files:
    img = cv2.imread(os.path.join(folder_path,file)) # Image to copy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # To grayscale
    cv2.imwrite(os.path.join(train_path,file), gray) # Save the image