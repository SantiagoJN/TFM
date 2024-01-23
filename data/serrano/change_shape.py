from PIL import Image
import os
import time
import datetime
from utils import create_safe_directory

# Original images
directory = "color_chnl_original/"
# Where we will save the images
output_directory = "color_chnl/"
create_safe_directory(output_directory)
# Desired size
size = (256, 256)

i = 0
file_list = os.listdir(directory)
total_imgs = len(file_list)
start_time = time.time()
# Iterate over all images
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load original image
        image = Image.open(os.path.join(directory, filename))

        # Transform it to desired size
        image = image.resize(size)

        # Save it
        image.save(os.path.join(output_directory, filename))
        
        # Some feedback
        if i%100 == 0 and i != 0:
            now = datetime.datetime.now()
            timestamp = now.strftime("%H:%M:%S")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            total_estimation = total_imgs * elapsed_time / i
            left_time = total_estimation - elapsed_time
            hours = int(left_time // 3600)
            minutes = int((left_time % 3600) // 60)
            seconds = left_time % 60
            
            print(f'{timestamp} - [{i}/{total_imgs}], approx {hours}:{minutes}:{seconds:.2f} left')
        i += 1

print("Done!")