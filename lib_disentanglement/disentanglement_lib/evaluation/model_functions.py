
import numpy as np
import torch
from torch import nn


# Function that takes a sample and outputs its representation
'''def representation_function(x):
    representations = np.zeros((x.shape[0], ground_truth_data.means.shape[1])) # Batch size x latent dim
    for sample in range(x.shape[0]): # For each sample of this batch
        # index = np.where(np.all(ground_truth_data.images == x[sample].squeeze(), axis=(1,2,3)))
        for image in range(ground_truth_data.images.shape[0]): # For each image of the dataset
        if np.all(ground_truth_data.images[image] == x[sample].squeeze(), axis=(0,1,2)):
            print(f'Found sample {sample} in image with index {image}!')
            representations[sample] = ground_truth_data.means[image] # Save its respective mean
            continue # There should be only one coincidence

    return representations
'''


def init_specific_model(model_type, img_size, latent_dim):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    encoder = get_encoder(model_type)
    decoder = get_decoder(model_type)
    model = VAE(img_size, encoder, decoder, latent_dim)
    model.model_type = model_type  # store to help reloading
    return model


class VAE(nn.Module):
    def __init__(self, img_size, encoder, decoder, latent_dim):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64], [256, 256], [512, 512]]:
            raise RuntimeError("{} sized images not supported. Only (None, 32, 32), (None, 64, 64), \
                                (None, 256x256) and (None, 512x512) \
                                supported. Build your own architecture or reshape images!".format(img_size))

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
        self.encoder = encoder(img_size, self.latent_dim)
        self.decoder = decoder(img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        # print(latent_dist[0][:5])
        
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)
        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample

def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {nn.LeakyReLU: "leaky_relu", nn.ReLU: "relu", nn.Tanh: "tanh",
              nn.Sigmoid: "sigmoid", nn.Softmax: "sigmoid"}
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(layer, activation="relu"):
    """Initialize a linear layer.
    Args:
        layer (nn.Linear): parameters to initialize.
        activation (`torch.nn.modules.activation` or str, optional) activation that
            will be used on the `layer`.
    """
    x = layer.weight

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity='leaky_relu')
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity='relu')
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))


def weights_init(module):
    if isinstance(module, torch.nn.modules.conv._ConvNd):
        # TO-DO: check litterature
        linear_init(module)
    elif isinstance(module, nn.Linear):
        linear_init(module)


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
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))
            #print(f'L4: {x.size()}')
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


# ALL decoders should be called Decoder<Model>
def get_decoder(model_type):
    model_type = model_type.lower().capitalize()
    return eval("Decoder{}".format(model_type))


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

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
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 256x256 do _more_ convolutions
        if self.img_size[1] == self.img_size[2] == 256:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.convT_32 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.convT_16 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        
        # If input image is 512x512 do _more_ convolutions
        if self.img_size[1] == self.img_size[2] == 512:
            self.convT_64 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.convT_32 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.convT_16 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            self.convT_8 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
            # self.convT_512 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        if self.img_size[1] == self.img_size[2] == 512:
            x = torch.relu(self.convT_64(x))
            x = torch.relu(self.convT_32(x))
            x = torch.relu(self.convT_16(x))
            x = torch.relu(self.convT_8(x))
            # x = torch.relu(self.convT_512(x))
        if self.img_size[1] == self.img_size[2] == 256:
            x = torch.relu(self.convT_64(x))
            x = torch.relu(self.convT_32(x))
            x = torch.relu(self.convT_16(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x
