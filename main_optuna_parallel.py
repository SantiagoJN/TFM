"""
    Adaptation of the training process to include the hyperparameter 
    optimization process using the Optuna library with parallelization: 
    https://optuna.readthedocs.io/en/stable/

    Using a file-based Optuna storage: https://tinyurl.com/59v8behm

"""
import optuna # For hyperparameter study
from optuna.storages import JournalStorage, JournalFileStorage # For file-based storage!
import joblib # For storing finished studies -> https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-save-and-resume-studies

import random 

import argparse
import logging
import sys
import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from configparser import ConfigParser

from torch import optim
import torch
import numpy as np
import time

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, DATASETS
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining
from pynvml import *

from torch.utils.tensorboard import SummaryWriter
import subprocess
from optuna.samplers import TPESampler


from datetime import datetime


CONFIG_FILE = "hyperparam.ini"
path_config = get_config_section([CONFIG_FILE], "Paths")
RES_DIR = path_config['results_dir']
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba", "best_dsprites"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]

# study = None # Global variable to be able to store it in the middle of the trial (?)

def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')
    general.add_argument('-f', '--frequency', type=int, default=default_config['frequency'],
                         help='How frequently to store individual traversals during learning (every X epochs). 0 to disable')
    general.add_argument('-t', '--trials', type=int, default=5,
                         help='How many trials we want to make during the hyperparameter optimization process')
    general.add_argument('-g', '--gpu', type=int, default=0,
                         help='Which GPU to use; 0-GF RTX 2080 Ti , 1-Quadro P5000')
    general.add_argument('-p', '--parallelization_file', type=str, default="undefined",
                         help='How to name the file used for the parallelization')
    general.add_argument('-n', '--node', type=int, default=99,
                         help='The node identifier') # Trying to fix the file not found error when training

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'], choices=DATASETS)
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')
    # * For the GECO implementation
    factor.add_argument('--max-dim', type=int,
                        default=default_config['max_dim'],
                        help="Initial maximum dimensions (GECO implementation).")
    factor.add_argument('--max-error', type=int,
                        default=default_config['max_error'],
                        help="Maximum error handled (GECO implementation).")
    factor.add_argument('--use-GECO', type=int,
                        default=default_config['use_GECO'],
                        help="Whether or not use the GECO implementation.")

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args


#* Function to store the study before all trials end
def store_best_trial(study_dir):
    trial = study.best_trial

    joblib.dump(study, study_dir)
    print(f'Saved study in {study_dir}')

def extract_float(text):
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


# global trial_num
# trial_num = 0
"""
    Function necessary for the procedure of searching the best hyperparameter
    configuration. It samples different hyperparameter values, and stores the
    configuration that provides best results; in this case, best vae_loss.
"""
def objective(trial, args, logger, stream, device, writer_dir, writer):
    # 1) Select specific arguments for this trial
    # TODO: Set the hyperparameters we want to study
    latent_dim = trial.suggest_int('latent_dim', 15, 25)
    # epochs = trial.suggest_int('epochs', 1000, 4000) 
    epochs = 4000 #! Fixed value !
    learning_rate = trial.suggest_float("learning_rate", 0.00005, 0.005, log=True)
    learning_rate_disc = trial.suggest_float("learning_rate_disc", 0.000001, 0.0001, log=True)
    gamma = trial.suggest_float("gamma", 6.0, 9.0)
    
    # Modify the program arguments to reflect the changes of this experiment
    args.latent_dim = latent_dim
    args.epochs = epochs
    args.lr = learning_rate
    args.lr_disc = learning_rate_disc
    args.factor_G = gamma

    args.batch_size = 700 # ! Set it to fill the available GPU(s) as much as possible

    exp_name = f'optuna/{args.name}_{epochs}_{latent_dim}_{gamma:.2f}_{learning_rate:.5f}_{learning_rate_disc:.6f}'
    exp_dir = os.path.join(RES_DIR, exp_name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    create_safe_directory(exp_dir, logger=logger)

    frequency_optuna = 400
    print(f'[INFO] Informing the Optuna pruner every {frequency_optuna} epochs')
    
    # 2) Train the model following the previous training process
    # PREPARES DATA
    train_loader = get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger)
    
    logger.info("Train {} with {} samples".format(args.dataset, len(train_loader.dataset)))

    compute_test_losses = False
    if args.dataset == "serrano": # Load the test dataset to compute its loss during training
        test_loader = get_dataloaders("serranotest",
                                    batch_size=args.batch_size,
                                    logger=logger)
        test_loss_f = get_loss_f(args.loss,
                        n_data=len(test_loader.dataset),
                        device=device,
                        **vars(args))
        compute_test_losses = False #! To tell the training process to store test losses
        if compute_test_losses:
            logger.info("Test {} with {} samples".format("serranotest", len(test_loader.dataset)))

    else:
        test_loader = None # Else, just don't declare it
        test_loss_f = None

    # PREPARES MODEL
    args.img_size = get_img_size(args.dataset)  # stores for metadata
    model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    logger.info('Num parameters in model: {}'.format(get_n_param(model)))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(device)  # make sure trainer and viz on same device
    gif_visualizer = None # Don't save anything, training gif is not so useful and slows down the training :/
    loss_f = get_loss_f(args.loss,
                        n_data=len(train_loader.dataset),
                        device=device,
                        **vars(args))
    trainer = Trainer(model, optimizer, loss_f,
                        device=device,
                        logger=logger,
                        save_dir=exp_dir,
                        is_progress_bar=not args.no_progress_bar,
                        gif_visualizer=gif_visualizer,
                        writer_dir=writer_dir,
                        writer=writer,
                        name=args.name)
    vae_loss = trainer(train_loader,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,
            test_loader=test_loader,
            compute_test_losses=compute_test_losses,
            test_loss_f=test_loss_f,
            trial_optuna=trial,
            frequency_optuna=frequency_optuna)

    # SAVE MODEL AND EXPERIMENT INFORMATION
    save_model(trainer.model, exp_dir, metadata=vars(args))

    # 2.5) Save study after each trial (although it is not updated with the values of this trial)
    if (trial.number > 20): # If we are in the first trials, there is nothing to save yet!
        curr_dt = datetime.now()
        rndm = f'{curr_dt.year}{curr_dt.month}{curr_dt.day}'
        study_dir = f"HPO_studies/study_{rndm}_{args.name}.pkl"
        store_best_trial(study_dir)
    # trial_num += 1

    # 3) Computing metrics
    print('######## Computing metrics ########')
    interpreter = "/mnt/cephfs/home/graphics/sjimenez/anaconda3/envs/env_metrics/bin/python3.7"
    script = "/mnt/cephfs/home/graphics/sjimenez/lib_disentanglement/disentanglement_lib/evaluation/custom_factor_vae_test.py"
    dataset = "masked-validation"
    num_trials = "5"
    cmd = f"{interpreter} {script} {exp_name} {dataset} {num_trials}"
    output = subprocess.check_output(cmd, shell=True)

    output = output.decode("utf-8")

    factorVAE_idx = output.find("FactorVAE:")
    MIS_idx = output.find("Mutual Info Score:")
    print(output)
    print("-----------------------------")
    fVAE = extract_float(output[factorVAE_idx+11:factorVAE_idx+16])
    MIS = extract_float(output[MIS_idx+19:MIS_idx+24])
    print(f'FVAE = {fVAE}, MIS = {MIS}')

    return fVAE + (1.0 - MIS)




def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    start_time = time.time()
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    device = torch.device('cuda:0') #get_device(is_gpu=not args.no_cuda)
    print(f'----Using {torch.cuda.get_device_name(0)}----')

    writer_root = path_config['runs_root']
    writer_dir = f"{writer_root}/{args.dataset}/{args.name}"
    writer = SummaryWriter(writer_dir)

    NAME = "Parallel6"
    storage = JournalStorage(JournalFileStorage(f"HPO_{NAME}.log"))
    print(f'[INFO] Using the file HPO_parallel_{args.parallelization_file} for communication between parallel processes')
    min_res = 100
    max_res = 4001
    red_factor = 4
    num_brackets = np.floor(np.emath.logn(red_factor,(max_res/min_res))) + 1
    print(f'[INFO] Using {num_brackets} brackets')

    global study # Make it globally visible to use the study inside the optimization procedure
    study = optuna.load_study(
        study_name=NAME, # ! Keep this coherent with create_study
        sampler=TPESampler(), 
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=min_res, max_resource=max_res, reduction_factor=red_factor #-> Slide 200 -> Lower it to 4 to let more trainings
        ),
        # direction='maximize', # A study is associated with a direction, we cannot re-define it when loading the study.
        storage=storage) # ! Keep this coherent with create_study) 

    # study = optuna.create_study(
    #     sampler=TPESampler(), 
    #     pruner=optuna.pruners.HyperbandPruner(
    #         min_resource=1000, max_resource=args.epochs, reduction_factor=8 #-> Slide 200 -> Lower it to 10 to let more trainings
    #     ),
    #     direction='maximize',
    #     storage=storage,
    #     load_if_exists=True)


    # Set a decent number of trials: 
    # If you use HyperbandPruner with TPESampler, it’s recommended to consider 
    # setting larger n_trials or timeout to make full use of the characteristics 
    # of TPESampler because TPESampler uses some (by default, 10) Trials for its startup.
    # ! for example, if HyperbandPruner has 4 pruners in it, at least 4x10 trials are consumed for startup.
    study.optimize(lambda trial: objective(trial, args, logger, stream, device, writer_dir, writer), n_trials=args.trials)

    trial = study.best_trial

    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
    # rndm = ''.join((random.choice('abcdxyzpqr') for i in range(5))) # Random string in case we mess up with names
    curr_dt = datetime.now()
    rndm = f'{curr_dt.year}{curr_dt.month}{curr_dt.day}' # Set the day of today (seed is fixed and rndm is not random ¬¬)
    study_dir = f"HPO_studies/FINALstudy_{rndm}_{NAME}.pkl"
    joblib.dump(study, study_dir)
    print(f'Saved study in {study_dir}')
        

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    print(f'[SUCCESS] Process done in {hours} hours, {minutes} minutes and {seconds:.2f} seconds.')
        
    writer.close() # Make sure it is closed properly 


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
    default_config = get_config_section([CONFIG_FILE], "Custom")
    if args.checkpoint_every == default_config["checkpoint_every"]: # If it is the default one
        args.checkpoint_every = np.floor(args.epochs/10)
        print(f'Saving checkpoints every {args.checkpoint_every} epochs.')
    main(args)
