# Datasets

During this internship, I've been working mainly with the [Serrano21 dataset](https://mig.mpi-inf.mpg.de/). To test different features, I modified the dataset in some ways, and the purpose of this readme is to explain what does each one mean. Also, in order to save as much space as possible, I have shared only the scripts needed to obtain each dataset, and here I will explain how we need to execute them to do so.

In the following sections I will explain how to obtain each dataset, but here is a summary of all of them:

 - `color_chnl_original`: The original dataset of Serrano21, containing 1024x1024 images.
 - `color_chnl`: The same dataset as the first one, but whose samples are downsized to 256x256. This will be used as starting point for the following datasets.
 - `balanced-glossiness`: A dataset that has a balanced number of samples for each level of glossiness. We made it because it seemed to be that the training dataset was slightly biased towards low glossiness, so this dataset has *n* samples per glossiness level. 

In the scripts that do not use the whole dataset, we are storing the indices of the samples that are in the new dataset. This is done mainly with the objective of running the TSNE script faster. In that script we need to access to the data and the different labels, and we can implement this part in a general way for all the datasets, and then filter the results using the indices that we are interested in.
 - `mini-serrano`: In this dataset we have the combinations between two geometries (sphere and blob), and three illuminations (art_studio, tiber and small_cathedral), which lead to 520 (BRDFs) * 2 (geometries) * 3 (illuminations) = 3120 samples.
 - `masked-serrano{1,2,3}`: Some datasets that have a mask applied to make the background black (although the illumination of the original sample still influences the appearance of the geometry).
 - `full-masked-serrano`: All the samples of the original dataset, without the background information.
 - `masked-{spheres,blobs,buddhas,dragons}`: Special datasets for each of those 4 geometries.
 - `gray-{spheres,blobs,buddhas,dragons}`: The same as the previous point, but in grayscale.

One can search the names of these datasets in the slides I made during the internship to get more information about the reasons and the results obtained with them.

## Main dataset

The Serrano21 dataset consists in 1024x1024 images containing renders of different combinations of geometries and illuminations. This resolution might be useful for some applications, but we are going to resize the samples to match our architecture (if needed, we can easily change this architecture to handle other resolutions, as explained in the [main readme](https://github.com/SantiagoJN/disentangling-vae)).

After downloading the original dataset from the [official page](https://mig.mpi-inf.mpg.de/), we are going to change the name of the folder that contains the images to the name `color_chnl_original`. Then, we will execute the script `change_shape.py`, and once it ends, we should see a new folder `color_chnl` that contains all the samples from the original dataset, but downsized to our desired shape (in this case, 256x256, but it can be easily changed in the script).

## Mini-serrano
To test if our model was able to disentangle a simpler dataset, we built mini-serrano, that has a reduced number of samples and geometries. To create it, we just need to run the `get_train_subset` script with the mode 1 (a variable inside the code).

## Masked datasets
Another way we found of decreasing the complexity of the dataset was to apply a mask to the samples, and get rid of the background information, which was adding a level of complexity that our VAE needed to learn. For that, we used one layer of the HDR information (HDR samples are in the `one_exr_per_geom` folder provided), and then we masked every sample with its own layer. Here we have again a **mode** to select which dataset we want to build:

 1. masked-serrano: spheres and blobs with all illuminations
 2. masked-serrano2: the same samples as mini-serrano, but masked
 3. masked-serrano3: spheres and cylinder with all illuminations
 4. full-masked-serrano: all the samples of the original dataset (masked)
 5. masked-spheres: all the samples with spheres masked
 6. masked-blobs: all the samples with blobs masked
 7. masked-buddhas: all the samples with buddhas masked
 8. masked-dragons: all the samples with dragons masked

## Pattern dataset
We also tested the influence of the background color of the images in the learnt latent space organization. To this end, we made a dataset that contained the same samples as masked-serrano, but in this case using a colored dataset as background (instead of the plain black color). This pattern is stored in a 256x256 image named `pattern.jpg`, but one can change it if necessary.

## Grayscale datasets
Decreasing the complexity once again, we can take the masked datasets we built before, and convert the samples to grayscale ones. With this, we want to test if our model is able to disentangle the illumination. Using again a **mode** variable, we can select which dataset we want to build:

 1. gray-spheres
 2. gray-blobs
 3. gray-buddhas
 4. gray-dragons

## Reading ground truth labels
I also included a small script named `read_xlsx.py`, which I have used to get the ground truth labels (glossiness, illumination...) of the different samples, used in the TSNE script. Currently it is saving the names of the names of the samples, but if we want to store, for example, the 4th column (the glossiness), then we just need to change the 17th line to `for col in  worksheet.iter_cols(4,4)`, also changing the name of the output file.
