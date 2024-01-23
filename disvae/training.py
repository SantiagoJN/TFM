import imageio
import logging
import os
from timeit import default_timer
from collections import defaultdict

from tqdm import trange
import torch
from torch.nn import functional as F

from utils.helpers import create_safe_directory

from disvae.utils.modelIO import save_model

import subprocess
import optuna

TRAIN_LOSSES_LOGFILE = "train_losses.log"


class Trainer():
    """
    Class to handle training of model.

    Parameters
    ----------
    model: disvae.vae.VAE

    optimizer: torch.optim.Optimizer

    loss_f: disvae.models.BaseLoss
        Loss function.

    device: torch.device, optional
        Device on which to run the code.

    logger: logging.Logger, optional
        Logger.

    save_dir : str, optional
        Directory for saving logs.

    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.

    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 gif_visualizer=None,
                 is_progress_bar=True,
                 writer_dir="Undefined",
                 writer=None,
                 name="no_name"):

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE))
        self.gif_visualizer = gif_visualizer
        self.logger.info("Training Device: {}".format(self.device))

        self.writer_dir = writer_dir
        self.writer = writer
        self.name = name

    def extract_float(self, text):
        result = ''
        found_dot = False

        for char in text:
            if char.isdigit():
                result += char
            elif char == '.' and not found_dot:
                result += char
                found_dot = True

        try:
            float_val = float(result)
            return float_val
        except ValueError:
            print(f'[ERROR] Invalid string to obtain a float: {text}')
            return None

    def __call__(self, data_loader,
                 epochs=10,
                 checkpoint_every=10,
                 test_loader=None,
                 compute_test_losses=False,
                 test_loss_f=None,
                 trial_optuna=None,
                 frequency_optuna=0):
        """
        Trains the model.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        epochs: int, optional
            Number of epochs to train the model for.

        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """
        start = default_timer()
        self.model.train()
        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,
                                                                               mean_epoch_loss))
            if trial_optuna is not None and (epoch+1) % frequency_optuna == 0: # We're performing HPO
                print(f"Informing optuna about the intermediate value at epoch {epoch+1}")

                #? Saving the model to disk so we can compute metrics
                exp_name = f'optuna/intermediate_model_{self.name}'
                exp_dir = f'results/{exp_name}'
                create_safe_directory(exp_dir)
                save_model(self.model, exp_dir)

                #? Compute the metrics that will be informed to the pruner
                interpreter = "/mnt/cephfs/home/graphics/sjimenez/anaconda3/envs/env_metrics/bin/python3.7"
                script = "/mnt/cephfs/home/graphics/sjimenez/lib_disentanglement/disentanglement_lib/evaluation/custom_factor_vae_test.py"
                dataset = "masked-validation"
                num_trials = "5"
                cmd = f"{interpreter} {script} {exp_name} {dataset} {num_trials}"
                output = subprocess.check_output(cmd, shell=True)

                output = output.decode("utf-8")

                factorVAE_idx = output.find("FactorVAE:")
                MIS_idx = output.find("Mutual Info Score:")
                fVAE = self.extract_float(output[factorVAE_idx+11:factorVAE_idx+16])
                MIS = self.extract_float(output[MIS_idx+19:MIS_idx+24])

                metric = fVAE + (1.0 - MIS)
                print(f'Epoch {epoch}; metric: {metric} (fVAE={fVAE}, MIS={MIS})')
                trial_optuna.report(metric, epoch) # !! Valor intermedio: la métrica que usemos en Optuna (Z-Max+MIS...)
                if trial_optuna.should_prune():
                    self.logger.info(f'[WARNING] Optuna pruning at epoch {epoch+1} with metric {metric}!')
                    raise optuna.TrialPruned()
            
            self.losses_logger.log(epoch, storer, self.writer)

            if self.gif_visualizer is not None:
                self.gif_visualizer()

            if epoch % checkpoint_every == 0:
                save_model(self.model, self.save_dir,
                           filename="model-{}.pt".format(epoch))
            
            if compute_test_losses:
                with torch.no_grad():
                    # At the end of the epoch, compute test losses using test dataset,
                    #   and store it to be displayed in tensorboard
                    storer_test = defaultdict(list)
                    #with torch.no_grad(): # Don't do any gradient for this :d
                    test_loss = 0
                    for _, (data, _) in enumerate(test_loader):
                        iter_loss = self._get_test_loss(data, storer_test, test_loss_f)
                        test_loss += iter_loss
                    mean_test_loss = test_loss / len(test_loader)
                    self.logger.info('Epoch: {} Average TEST loss per image: {:.2f}'.format(epoch + 1,
                                                                                mean_test_loss))
                    self.writer.add_scalar("test_loss", mean_test_loss, epoch) # k is the identifier

        # ? Not saving training.gif anymore ?
        # if self.gif_visualizer is not None:
        #     self.gif_visualizer.save_reset()
        print('training.gif not saved~~')

        self.model.eval()

        delta_time = (default_timer() - start) / 60
        self.logger.info('Finished training after {:.1f} min.'.format(delta_time))
        return mean_epoch_loss

    def _train_epoch(self, data_loader, storer, epoch):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        storer: dict
            Dictionary in which to store important variables for vizualisation.

        epoch: int
            Epoch number

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                      disable=not self.is_progress_bar)
        with trange(len(data_loader), **kwargs) as t:
            for _, (data, labels) in enumerate(data_loader):
                iter_loss = self._train_iteration(data, storer, labels)
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(data_loader)
        return mean_epoch_loss

    def _train_iteration(self, data, storer, labels):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data: torch.Tensor
            A batch of data. Shape : (batch_size, channel, height, width).
        
        labels: torch.Tensor
            The labels of the previous data. Shape : (batch_size, 1).

        storer: dict
            Dictionary in which to store important variables for vizualisation.
        """
        batch_size, channel, height, width = data.size()
        data = data.to(self.device)
        labels = labels.to(self.device)

        try:
            recon_batch, latent_dist, latent_sample = self.model(data)
            loss = self.loss_f(data, recon_batch, latent_dist, self.model.training,
                               storer, latent_sample=latent_sample)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        except ValueError:
            # for losses that use multiple optimizers (e.g. Factor)
            loss = self.loss_f.call_optimize(data, self.model, self.optimizer, storer, labels)

        return loss.item()
    
    def _get_test_loss(self, data, storer_test, test_loss_f):
        """
        Based on the _train_iteration() function above.
        Just computes the test losses using the current model. 
        """
        data = data.to(self.device)
        loss = test_loss_f.compute_test_loss(data, self.model, self.optimizer, storer_test)

        return loss.item()



class LossesLogger(object):
    """Class definition for objects to write data to log files in a
    form which is then easy to be plotted.
    """

    def __init__(self, file_path_name):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger("losses_logger")
        self.logger.setLevel(1)  # always store
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(1)
        self.logger.handlers.clear() # To avoid overwriting previously-opened loggers (HPO trains multiple models)
        self.logger.addHandler(file_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer, writer):
        """Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)

            #print(f'Epoch: {epoch}, k: {k}, mean v: {mean(v)}')
            writer.add_scalar(k, mean(v), epoch) # k is the identifier


# HELPERS
def mean(l):
    """Compute the mean of a list"""
    return sum(l) / len(l)
