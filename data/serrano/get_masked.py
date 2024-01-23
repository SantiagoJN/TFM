import numpy as np
import OpenEXR as exr
import Imath
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageChops
import shutil
import os
from utils import create_safe_directory

# With this script we can create a number of datasets:
#   1: masked-serrano
#   2: masked-serrano2
#   3: masked-serrano3
#   4: full-masked-serrano
#   5: masked-spheres
#   6: masked-blobs
#   7: masked-buddhas
#   8: masked-dragons

mode = 8

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


def meet_condition(value, mode):
    if mode == 1: # masked-serrano
        return "sphere" in value or "blob" in value
    elif mode == 2: # masked-serrano2
        return ("sphere" in value or "blob" in value) and ("studio" in value or "tiber" in value or "cathedral" in value)
    elif mode == 3: # masked-serrano3
        return ("sphere" in value or "cylinder" in value)
    elif mode == 4: # full-masked-serrano
        return True
    elif mode == 5: # masked-spheres
        return "sphere" in value
    elif mode == 6: # masked-blobs
        return "blob" in value
    elif mode == 7: # masked-buddhas
        return "happy" in value
    elif mode == 8: # masked-dragons
        return "xyzrgb" in value
    else:
        print(f"[ERROR] Invalid mode: {mode}")
        exit()

# Source folder
folder_path = "color_chnl"

# Destiny folder
if mode == 1:
    train_path = "masked-serrano"
elif mode == 2:
    train_path = "masked-serrano2"
elif mode == 3:
    train_path = "masked-serrano3"
elif mode == 4:
    train_path = "full-masked-serrano"
elif mode == 5:
    train_path = "masked-spheres"
elif mode == 6:
    train_path = "masked-blobs"
elif mode == 7:
    train_path = "masked-buddhas"
elif mode == 8:
    train_path = "masked-dragons"
else:
    print(f"[ERROR] Invalid mode: {mode}")
    os.exit()

create_safe_directory(train_path)

# All files
files = os.listdir(folder_path)

######################################
######################################

print('Reading EXR files...')
blob_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][blob][MERL-alum-bronze].exr")
sphere_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][sphere][MERL-alum-bronze].exr")
bunny_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][bunny_vt][MERL-alum-bronze].exr")
cylinder_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][cylinder][MERL-alum-bronze].exr")
happy_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][happy_vrip_vt][MERL-alum-bronze].exr")
statuette_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][statuette_vt][MERL-alum-bronze].exr")
surface2_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][surface2-1000x1000part][MERL-alum-bronze].exr")
teapot_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][teapot_vt_scale][MERL-alum-bronze].exr")
dragon_mask = readEXR("one_exr_per_geom/[1_1_cambridge_2k][xyzrgb_dragon_vt][MERL-alum-bronze].exr")
# mask_np = np.array(mask_pil)
# mask_np = (mask_np > 250).astype('uint8') * 255


# plt.imshow(blob_mask)
# plt.show()

# plt.imshow(sphere_mask)
# plt.show()

length = len(files)

i = 0
for file in files:
    if i%100 == 0:
        print(f'[{i}/{length}]')
    
    if meet_condition(file, mode): # If the file meets the condition, apply the mask and save it
        #if "studio" in file or "tiber" in file or "cathedral" in file:
        image = Image.open(os.path.join(folder_path, file))
        if "sphere" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), sphere_mask)
        elif "blob" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), blob_mask)
        elif "bunny" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), bunny_mask)
        elif "cylinder" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), cylinder_mask)
        elif "happy" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), happy_mask)
        elif "statuette" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), statuette_mask)
        elif "surface2" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), surface2_mask)
        elif "teapot" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), teapot_mask)
        elif "xyzrgb" in file:
            img_masked = ImageChops.composite(image, Image.new('RGB', image.size, (0,0,0)), dragon_mask)
        else:
            print(f'Something went wrong with {file}')
            os.exit()

        img_masked.save(os.path.join(train_path, file)) # Save the image

    i += 1
