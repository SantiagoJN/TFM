import numpy as np
import OpenEXR as exr
import Imath
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageChops
import PIL.ImageOps   
import shutil
import os
from utils import create_safe_directory

def readEXR(filename):
    """Read color + depth data from EXR image file.
    
    Parameters
    ----------
    filename : str
        File path.
        
    Returns
    -------
    img : RGB or RGBA image in float32 format. Each color channel
          lies within the interval [0, 1].
          Color conversion from linear RGB to standard RGB is performed
          internally. See https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
          for more information.
          
    Z : Depth buffer in float32 format or None if the EXR file has no Z channel.
    """
    
    exrfile = exr.InputFile(filename)
    header = exrfile.header()
    
    dw = header['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channelData = dict()
    
    # convert all channels in the image to numpy arrays
    for c in header['channels']:
        # print(c)
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
    
    # print(header['channels'])
    colorChannels = ['albedo.R', 'albedo.G', 'albedo.B', 'albedo.A'] if 'albedo.A' in header['channels'] else ['albedo.R', 'albedo.G', 'albedo.B']
    img = np.concatenate([channelData[c][...,np.newaxis] for c in colorChannels], axis=2)
    
    # linear to standard RGB
    img[..., :3] = np.where(img[..., :3] <= 0.0031308,
                            12.92 * img[..., :3],
                            1.055 * np.power(img[..., :3], 1 / 2.4) - 0.055)
    
    # sanitize image to be in range [0, 1]
    img = np.where(img < 0.0, 0.0, np.where(img > 1.0, 1, img))
    
    #Z = None if 'Z' not in header['channels'] else channelData['Z']
    
    img = Image.fromarray(np.uint8(img*255))
    img = img.resize((256,256))
    img = img.convert('1')

    return img

# Source folder
folder_path = "color_chnl"

# Destiny folder
train_path = "pattern-serrano"
create_safe_directory(train_path)

# Listing all images
files = os.listdir(folder_path)

######################################
######################################

print("Reading EXR files...")
blob_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][blob][MERL-alum-bronze].exr")
sphere_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][sphere][MERL-alum-bronze].exr")


pattern = Image.open("pattern.jpg")
# plt.imshow(pattern)
# plt.show()

length = len(files)

i = 0
for file in files:
    if i%100 == 0:
        print(f'[{i}/{length}]')
    if "studio" in file or "tiber" in file or "cathedral" in file:
        image = Image.open(os.path.join(folder_path, file))
        if "sphere" in file:
            final_img = Image.composite(image, pattern, sphere_mask)
            final_img.save(os.path.join(train_path, file))
        elif "blob" in file:
            final_img = Image.composite(image, pattern, blob_mask)
            final_img.save(os.path.join(train_path, file))
            

    i += 1
