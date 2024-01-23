# Program to plot in the console the name of 49 materials, which correspond to
#   7 different materials per glossiness levels. This is done to later build a 
#   dataset whose samples have a balanced glossiness level.

import os
import shutil
import numpy as np
import random
from utils import create_safe_directory

glossiness_levels = np.load("labels/glossiness_labels.npy")
names = np.load("labels/samples_names.npy")

create_safe_directory("balanced-glossiness")
num_samples = 500

used_BRDFs = []
for gloss_level in range(1,8):
    print(f'Glossiness {gloss_level}...')
    # Get the indices that have this glossiness
    querying_glossiness = [idx for idx,value in enumerate(glossiness_levels) if value==gloss_level] 
    valid_names = names[querying_glossiness] # Names of materials with certain glossiness

    #print(f"There are {len(querying_glossiness)} samples with gloss = {gloss_level}")
    used_BRDFs.append(random.sample(list(valid_names), num_samples)) # Print 500 random samples

used_BRDFs = np.array(used_BRDFs)
used_BRDFs = used_BRDFs.flatten()
indices = []
for sample in used_BRDFs: # Here we have the name of the images with balanced glossiness
    shutil.copy(os.path.join("color_chnl", sample), "balanced-glossiness")
    idx = np.where(names==sample)
    indices.append(idx[0])

indices = np.array(indices)
np.save("balanced-glossiness_idx", indices)


used_BRDFs = np.array(used_BRDFs)
used_BRDFs = used_BRDFs.flatten()
used_BRDFs_filtered = [(name.split("[")[3]).split("]")[0] for name in used_BRDFs]

BRDFs_set = set(used_BRDFs_filtered)

if len(used_BRDFs_filtered) != len(BRDFs_set):
    print("\n\n----There are some duplicated elements in the list")
else:
    print("\n\n----Created list with no duplicated elements")