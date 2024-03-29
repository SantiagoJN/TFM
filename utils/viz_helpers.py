import random

import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import torch
import imageio

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils.datasets import get_dataloaders
from utils.helpers import set_seed

from utils.helpers import get_config_section
CONFIG_FILE = "hyperparam.ini"
configuration = get_config_section([CONFIG_FILE], "Testing")


FPS_GIF = 12


def get_samples(dataset, num_samples, idcs=[], random_ids=True):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.

    num_samples : int, optional
        The number of samples to load from the dataset

    idcs : list of ints, optional
        List of indices to of images to put at the begning of the samples.
    """
    data_loader = get_dataloaders(dataset,
                                  batch_size=1,
                                  shuffle=idcs is None)
    if random_ids:
        idcs += random.sample(range(len(data_loader.dataset)), num_samples - len(idcs))
    else:
        #idcs += [id for id in range(num_samples)]
        '''idcs = [549519, 551563, 89908, 462936, 334007, 577322, 193003, 50693, \
                147809, 186307, 662129, 607420, 258809, 435979, 459829, 509477, \
                329808, 555245, 60378, 307811, 382826, 70201, 690960, 41923, \
                399868, 16675, 598939, 643688, 345789, 726430, 301428, \
                486612, 60888, 315665, 355717, 141597, 374466, 64090, 305218, \
                484373, 554080, 717385, 716945, 88591, 312173, 209406, 445680, \
                656682, 218538, 202032, 252609, 202459, 589042, 585426, 371783, \
                17575, 641292, 455927, 271736, 570746, 635554, 127508, 697991, \
                230527, 18144, 673912, 134022, 428132, 513737, 695830, 616154, \
                430584, 533, 611319, 397372, 206278, 241743, 544118, 253328, \
                594344, 560339, 454024, 322447, 54661, 567234, 276214, 403670, \
                238274, 316431, 635165, 7323, 682309, 535499, 261826, 353631, \
                723932, 625106, 287873, 19375, 178749]'''
        #idcs = [181291, 388773, 263017, 329025, 144108, 104054, 432577, 685166, 126172, 716220, 101733, 146444, 469693, 426172, 641091, 269038, 37403, 597968, 98421, 334200, 142819, 378501, 701322, 248193, 466652, 707652, 717065, 75700, 561573, 49471, 314725, 516500, 130159, 505300, 328458, 121000, 571874, 304134, 649366, 24067, 456666, 568716, 355380, 255117, 60806, 5857, 112599, 170474, 182699, 610182, 314233, 83836, 298730, 277046, 546187, 234009]
        #idcs = [62, 2290, 580, 1226, 697, 1271, 1823, 773, 1798, 132, 810, 3017, 1400, 458, 517, 2787, 2265, 1009, 681, 1569, 1261, 366, 756, 1570, 2856, 190, 1678, 68, 627, 639, 409, 795, 1632, 2166, 3070, 2098, 59, 249, 618, 2128, 690, 1172, 2245, 587, 2224, 2879, 2607, 1194, 2318, 1541, 1092, 1243, 3082, 959, 554, 2969]
        idcs = configuration['idcs']
    # for idx in idcs:
    #     print(f'Indice {idx}')
    #     plt.imshow(np.swapaxes(np.swapaxes(data_loader.dataset[idx][0],0,2),0,1))
    #     plt.show()
    samples = torch.stack([data_loader.dataset[i][0] for i in idcs], dim=0)
    print("Selected idcs: {}".format(idcs))

    return samples

def get_dataset(dataset, name, results_path):
    """ Generate a number of samples from the dataset.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    """

    print(f'Loading the dataset {name}...')
    data_loader = get_dataloaders(dataset,
                                  batch_size=1,
                                  shuffle=False)
    num_samples = len(data_loader)
    samples = torch.stack([data_loader.dataset[i][0] for i in range(num_samples)], dim=0)

    #? Reordered so that the channels are at the end (to respect the order in disentanglement_lib)
    samples_store = np.swapaxes(np.swapaxes(samples.detach().numpy(),1,2),2,3)
    # for image in range(3):
    #     plt.imshow(np.swapaxes(np.swapaxes(samples[image].detach().numpy(),0,2),0,1))
    #     plt.show()

    data_path = f'{results_path}/dataset'
    print(f'Storing the dataset with shape {samples_store.shape} to {data_path}')
    np.save(data_path, samples_store)
        
    return samples.numpy()



def sort_list_by_other(to_sort, other, reverse=True):
    """Sort a list by an other."""
    # print(to_sort)
    # print(other)
    return [el for _, el in sorted(zip(other, to_sort), reverse=reverse)]


# TO-DO: clean
def read_loss_from_file(log_file_path, loss_to_fetch):
    """ Read the average KL per latent dimension at the final stage of training from the log file.
        Parameters
        ----------
        log_file_path : str
            Full path and file name for the log file. For example 'experiments/custom/losses.log'.

        loss_to_fetch : str
            The loss type to search for in the log file and return. This must be in the exact form as stored.
    """
    EPOCH = "Epoch"
    LOSS = "Loss"

    logs = pd.read_csv(log_file_path)

    df_last_epoch_loss = logs[logs.loc[:, EPOCH] == logs.loc[:, EPOCH].max()]
    # print(logs.loc[:, EPOCH])
    df_last_epoch_loss = df_last_epoch_loss.loc[df_last_epoch_loss.loc[:, LOSS].str.startswith(loss_to_fetch), :]
    # ? Modificación para quitar un warning de future :/
    # https://pandas.pydata.org/docs/dev/whatsnew/v1.5.0.html#inplace-operation-when-setting-values-with-loc-and-iloc
    df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch, "").astype(int)

    # print(df_last_epoch_loss)
    # print('--------------')
    # df_last_epoch_loss.loc[:, LOSS] = df_last_epoch_loss.loc[:, LOSS].str.replace(loss_to_fetch, "")
    # print(df_last_epoch_loss)
    # exit()


    df_last_epoch_loss = df_last_epoch_loss.sort_values(LOSS).loc[:, "Value"]
    return list(df_last_epoch_loss)


def add_labels(input_image, labels):
    """Adds labels next to rows of an image.

    Parameters
    ----------
    input_image : image
        The image to which to add the labels
    labels : list
        The list of labels to plot
    """
    new_width = input_image.width + 100
    new_size = (new_width, input_image.height)
    new_img = Image.new("RGB", new_size, color='white')
    new_img.paste(input_image, (0, 0))
    draw = ImageDraw.Draw(new_img)

    for i, s in enumerate(labels):
        draw.text(xy=(new_width - 100 + 0.005,
                      int((i / len(labels) + 1 / (2 * len(labels))) * input_image.height)),
                  text=s,
                  fill=(0, 0, 0))

    return new_img


def make_grid_img(tensor, **kwargs):
    """Converts a tensor to a grid of images that can be read by imageio.

    Notes
    -----
    * from in https://github.com/pytorch/vision/blob/master/torchvision/utils.py

    Parameters
    ----------
    tensor (torch.Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
        or a list of images all of the same size.

    kwargs:
        Additional arguments to `make_grid_img`.
    """
    grid = make_grid(tensor, **kwargs)
    img_grid = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    img_grid = img_grid.to('cpu', torch.uint8).numpy()
    return img_grid


def get_image_list(image_file_name_list):
    image_list = []
    for file_name in image_file_name_list:
        image_list.append(Image.open(file_name))
    return image_list


def arr_im_convert(arr, convert="RGBA"):
    """Convert an image array."""
    return np.asarray(Image.fromarray(arr).convert(convert))


def plot_grid_gifs(filename, grid_files, pad_size=7, pad_values=255):
    """Take a grid of gif files and merge them in order with padding."""
    grid_gifs = [[imageio.mimread(f) for f in row] for row in grid_files]
    n_per_gif = len(grid_gifs[0][0])

    # convert all to RGBA which is the most general => can merge any image
    imgs = [concatenate_pad([concatenate_pad([arr_im_convert(gif[i], convert="RGBA")
                                              for gif in row], pad_size, pad_values, axis=1)
                             for row in grid_gifs], pad_size, pad_values, axis=0)
            for i in range(n_per_gif)]

    imageio.mimsave(filename, imgs, fps=FPS_GIF)


def concatenate_pad(arrays, pad_size, pad_values, axis=0):
    """Concatenate lsit of array with padding inbetween."""
    pad = np.ones_like(arrays[0]).take(indices=range(pad_size), axis=axis) * pad_values

    new_arrays = [pad]
    for arr in arrays:
        new_arrays += [arr, pad]
    new_arrays += [pad]
    return np.concatenate(new_arrays, axis=axis)
