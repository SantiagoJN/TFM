"""
Module containing the encoders.
"""
import numpy as np

import torch
import os
from torch import nn


# ALL encoders should be called Enccoder<Model>
def get_encoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Encoder{}".format(model_type))


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel), (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for 10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 1024x1024, do some more convolutions
        if self.img_size[1] == self.img_size[2] == 1024:
            self.batch1 = nn.BatchNorm2d(hid_channels)
            self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.conv_128 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_32 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_16 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_8 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.batch2 = nn.BatchNorm2d(hid_channels)
            self.max2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # If input image is 512x512, do some more convolutions
        if self.img_size[1] == self.img_size[2] == 512:
            self.batch1 = nn.BatchNorm2d(hid_channels)
            self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_32 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_16 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_8 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.batch2 = nn.BatchNorm2d(hid_channels)
            self.max2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            # self.conv_512 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 256x256, do some more convolutions
        if self.img_size[1] == self.img_size[2] == 256:
            self.batch1 = nn.BatchNorm2d(hid_channels)
            self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_32 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_16 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.batch2 = nn.BatchNorm2d(hid_channels)
            self.max2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # If input image is 128x128, do some more convolutions
        if self.img_size[1] == self.img_size[2] == 128:
            self.batch1 = nn.BatchNorm2d(hid_channels)
            self.max1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
            self.conv_32 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.conv_16 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.batch2 = nn.BatchNorm2d(hid_channels)
            self.max2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        #print(f'L1: {x.size()}')
        x = torch.relu(self.conv2(x))
        #print(f'L2: {x.size()}')
        x = torch.relu(self.conv3(x))
        #print(f'L3: {x.size()}')
        # print(f'================={self.img_size}')
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
            #print(f'L4: {x.size()}')
        if self.img_size[1] == self.img_size[2] == 128:
            x = self.batch1(x)
            x = torch.relu(self.conv_32(x))
            x = torch.relu(self.conv_16(x))
            x = self.batch2(x)
        if self.img_size[1] == self.img_size[2] == 256:
            x = self.batch1(x)
            #print(f'batch1: {x.size()}')
            x = torch.relu(self.conv_64(x))
            #print(f'L4: {x.size()}')
            x = torch.relu(self.conv_32(x))
            #print(f'L5: {x.size()}')
            x = torch.relu(self.conv_16(x))
            #print(f'L6: {x.size()}')
            x = self.batch2(x)
            #print(f'batch2: {x.size()}')
            # x = torch.relu(self.conv_512(x))
        if self.img_size[1] == self.img_size[2] == 512:
            x = self.batch1(x)
            #print(f'batch1: {x.size()}')
            x = torch.relu(self.conv_64(x))
            #print(f'L4: {x.size()}')
            x = torch.relu(self.conv_32(x))
            #print(f'L5: {x.size()}')
            x = torch.relu(self.conv_16(x))
            #print(f'L6: {x.size()}')
            x = torch.relu(self.conv_8(x))
            #print(f'L7: {x.size()}')
            x = self.batch2(x)
            #print(f'batch2: {x.size()}')
            # x = torch.relu(self.conv_512(x))
        if self.img_size[1] == self.img_size[2] == 1024:
            x = self.batch1(x)
            # print(f'batch1: {x.size()}')
            x = torch.relu(self.conv_128(x))
            # print(f'L4: {x.size()}')
            x = torch.relu(self.conv_64(x))
            # print(f'L5: {x.size()}')
            x = torch.relu(self.conv_32(x))
            # print(f'L6: {x.size()}')
            x = torch.relu(self.conv_16(x))
            # print(f'L7: {x.size()}')
            x = torch.relu(self.conv_8(x))
            # print(f'L8: {x.size()}')
            x = self.batch2(x)
            # print(f'batch2: {x.size()}')
            #x = torch.relu(self.conv_512(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        #print(f'view: {x.size()}')
        x = torch.relu(self.lin1(x))
        #print(f'Lin1: {x.size()}')
        x = torch.relu(self.lin2(x))
        #print(f'Lin2: {x.size()}')

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar
