

########################################################################################
#######Script we customized to test the factorVAE metric with our trained models.#######
#######                           Master thesis, MRGCV                           #######
########################################################################################


"""Tests for factor_vae.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl.testing import absltest
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore the spamming of warnings~~
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import serrano
from disentanglement_lib.data.ground_truth import subias
from disentanglement_lib.data.ground_truth import analytic
from disentanglement_lib.data.ground_truth import wild
from disentanglement_lib.data.ground_truth import tests

from disentanglement_lib.evaluation.metrics import custom_factor_vae, beta_vae, irs, custom_zmax
from disentanglement_lib.evaluation.metrics import factor_vae, dci, mig, mir, modularity_explicitness, sap_score
from disentanglement_lib.evaluation.metrics import unsupervised_metrics

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import model_functions 
import gin.tf

import json

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # -> Select GF RTX 2080 Ti
                                        # "1" -> Quadro P5000


PARAM_model_name = ""
PARAM_data_name = ""
PARAM_trials = 0
SAVE_RESULTS = True
in_HPO = False

class FactorVaeTest(absltest.TestCase):

  def test_metric(self):
        
    # Function that takes a sample and outputs its representation
    def representation_function(x):
        torch.cuda.empty_cache()

        x = torch.from_numpy(x) # Convert it to a torch tensor
        x = x.to(device) # Send it to the device
        x = x.float() # Converting it into float because otherwise it outputs errors

        x = x.squeeze() # Modify the shapes to fit it into the network
        x = np.swapaxes(np.swapaxes(x,1,2),1,3)
        #https://github.com/cezannec/capsule_net_pytorch/issues/4
        x = x.contiguous() # Solved the problem of "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)"

        # Introduce every image to the encoder and check that the output is 20D
        with torch.no_grad():
            mean, variance = model.encoder(x) # Finally get the latent representation

        mean = mean.to(torch.device("cpu")) # Send it back to the cpu
        mean = mean.detach().numpy()

        return mean
    
    def _identity_discretizer(target, num_bins):
      del num_bins
      return target
    
    # They use this discretizer in mig.gin
    def _histogram_discretize(target, num_bins=gin.REQUIRED):
        """Discretization based on histograms."""
        discretized = np.zeros_like(target)
        for i in range(target.shape[0]):
            discretized[i, :] = np.digitize(target[i, :], np.histogram(
                target[i, :], num_bins)[1][:-1])
        return discretized

    # ! ### Dataset ###
    # Current datasets: 
    #   + test-analytic
    #   + test-materials
    #   + test-illuminations
    #   + medium-hard
    #   + hard

    compute_supervised = True
    # perceptual_factor = 0:glossiness, 1:lightness, 2:anisotropy
    # dataset = 0:whole masked-serrano, 1:masked-serrano-train, 2:masked-serrano-test

    if PARAM_data_name != "":
        data_name = PARAM_data_name
    else:
        data_name = "masked-hpo-test"


    if data_name == "dumbdata":
        ground_truth_data = dummy_data.IdentityObservationsData()
    elif data_name == "dsprites":
        ground_truth_data = dsprites.DSprites()
    elif data_name == "serrano":
        ground_truth_data = serrano.Masked_Serrano()
    # Be careful when using these perceptual datasets, since they may interfere with the sample_observations_from_factors_no_color()
    elif data_name == "serrano_perceptual_gloss": 
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=0, perceptual_factor=0)
    elif data_name == "serrano_perceptual_lightness": 
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=0, perceptual_factor=1)
    elif data_name == "serrano_perceptual_gloss_TEST_medium" or data_name == "medium-hard": 
        ground_truth_data = serrano.Masked_Serrano_Perceptual_Gloss_TEST_medium()
    elif data_name == "serrano_perceptual_all": 
        ground_truth_data = serrano.Masked_Serrano_Perceptual_ALL()

    elif data_name == "subias":
        ground_truth_data = subias.Subias()
    elif data_name == "wild" or data_name == "hard":
        ground_truth_data = wild.Wild()

    elif data_name == "analytic_color" or data_name == "medium-easy":
        ground_truth_data = analytic.Analytic_Colored()

    elif data_name == "test-analytic":
        ground_truth_data = tests.Test_Analytic()
    elif data_name == "test-materials":
        ground_truth_data = tests.Test_Mat(selected_labels=selected_labels)
    elif data_name == "test-illuminations":
        ground_truth_data = tests.Test_Illum(selected_labels=selected_labels)

    elif data_name == "test-mat_ill":
        ground_truth_data = tests.Mat_Ill()
    elif data_name == "test-easy":
        ground_truth_data = tests.Easy()
    elif data_name == "test-train3":
        ground_truth_data = tests.Train()
        
    elif data_name == "masked-train-gloss":
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=1, perceptual_factor=0)
    elif data_name == "masked-test-gloss" or data_name == "serrano_perceptual_gloss_TEST" or data_name == "easy":
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=2, perceptual_factor=0)
    elif data_name == "masked-train-light":
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=1, perceptual_factor=1)
    elif data_name == "masked-test-light":
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=2, perceptual_factor=1)

    elif data_name == "masked-validation":
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=3, perceptual_factor=0)
    elif data_name == "masked-hpo-test":
        ground_truth_data = serrano.Masked_Serrano_Perceptual(dataset=4, perceptual_factor=0)

    elif data_name == "analytic_single":
        ground_truth_data = analytic.Analytic_Single_Color()
    else:
        print(f"[ERROR] Wrong data_name specified: {data_name}.")
        exit()
  
    # ! ### Pre-trained model ###
    model = 7
    if PARAM_model_name != "": # Not initialized
        print(f'Model name already defined: {PARAM_model_name}')
        model_name = PARAM_model_name
    elif model == 1:
        model_name = "factor_serrano_test59_0.0005_0.00001_128_100_20"
    elif model == 2:
        model_name = "factor_serrano_test60_0.0005_0.00001_128_500_20"
    elif model == 3:
        model_name = "factor_serrano_test61_0.0005_0.00001_128_1000_20"
    elif model == 4:
        model_name = "factor_serrano_test62_0.0005_0.00001_128_3500_20"
    elif model == 5:
        model_name = "factor_serrano_test76_0.0005_0.00001_128_2.4k_20_VSS1"
    elif model == 6:
        model_name = "factor_serrano_test80_0.0005_0.00001_128_4500_20_VSS4"
    elif model == 7:
        model_name = "factor_serrano_REPORT"
    else:
        print(f'[ERROR] Wrong model specified: {model}')
        exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = (f"/mnt/cephfs/home/graphics/sjimenez/disentangling-vae-master/results/{model_name}/model.pt")

    if PARAM_trials != 0:
        num_trials = PARAM_trials
    else:
        num_trials = 1 # ! Number of trials to avoid noise

    print(f'[INFO] Computing metrics [INFO] \n\tModel: {model_name}\n\tDataset: {ground_truth_data}')
                  
    img_size = (3, 256, 256)
    f = open(f"/mnt/cephfs/home/graphics/sjimenez/disentangling-vae-master/results/{model_name}/specs.json")
    hyperparams = json.load(f)
    latent_dim = hyperparams['latent_dim'] # 20 # ground_truth_data.means.shape[1]  # ? Dimensions of the latent space
    model_type = 'Burgess'
    model = model_functions.init_specific_model(model_type, img_size, latent_dim).to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    
    random_state = np.random.RandomState(1)
    scores_dict = {key:[] for key in ["factorvae", "zmax", "mig", "mir", "GTC", "GWC", "MIS"]}

    for trial in range(num_trials):
        print(f'============== TRIAL {trial} ==============')
        if compute_supervised and data_name != "hard" and data_name != "test-easy" and data_name != "test-train3" and data_name != "test-mat_ill":
            # ? ----------------------BetaVAE
            # scores = beta_vae.compute_beta_vae_sklearn(
            #     ground_truth_data, representation_function, random_state, None, 5, 2000, 2000)
            # scores_dict["betavae"] = scores["eval_accuracy"]

            # ? ----------------------FactorVAE
            scores = custom_factor_vae.compute_factor_vae( # It does the same as the original factor_vae
                ground_truth_data, representation_function, random_state, None, 5, 3000, 2000, 500) # Keep an eye on the num_variance_estimate
            scores_dict["factorvae"].append(scores["eval_accuracy"])

            if not in_HPO:
                # ? ----------------------Z_MAX
                scores = custom_zmax.compute_zmax( # The _implemented_ score
                    ground_truth_data, representation_function, random_state, None, 5, 3000, 2000, 500)
                scores_dict["zmax"].append(scores["eval_accuracy"])


                # ? ----------------------MIG
                gin.bind_parameter("discretizer.discretizer_fn", _histogram_discretize)
                gin.bind_parameter("discretizer.num_bins", 10)
                scores = mig.compute_mig(
                    ground_truth_data, representation_function, random_state, None, 3000) #3000
                scores_dict["mig"].append(scores["discrete_mig"])
                

                # ? ----------------------MIR
                gin.bind_parameter("discretizer.discretizer_fn", _histogram_discretize)
                gin.bind_parameter("discretizer.num_bins", 10)
                scores = mir.compute_mir(
                    ground_truth_data, representation_function, random_state, None, 3000) #3000
                scores_dict["mir"].append(scores["discrete_mir"])

        # ? ----------------------Unsupervised
        gin.bind_parameter("discretizer.discretizer_fn", _histogram_discretize)
        # * Bins = number of parts in which  we divide our space
        # https://www.geeksforgeeks.org/discretization-by-histogram-analysis-in-data-mining/
        gin.bind_parameter("discretizer.num_bins", 20) # ? <------- Keep in mind this parameter
        score = unsupervised_metrics.unsupervised_metrics(ground_truth_data, 
                    representation_function, random_state, None, 3000)
        scores_dict["GTC"].append(score["gaussian_total_correlation"])
        scores_dict["GWC"].append(score["gaussian_wasserstein_correlation"])
        scores_dict["MIS"].append(score["mutual_info_score"])

    print('****************************************')
    print('**************** SCORES ****************')
    print('****************************************')
    # print(scores_dict)
    if compute_supervised and data_name != "hard" and data_name != "test-easy" and data_name != "test-train3" and data_name != "test-mat_ill":
        print(f'--Supervised--')
        # print(f'\tBetaVAE: {scores_dict["betavae"]}')
        print(f'\tFactorVAE: {np.mean(scores_dict["factorvae"])}')
        if not in_HPO:
            print(f'\tZ-Max: {np.mean(scores_dict["zmax"])}')
            print(f'\tMIG: {np.mean(scores_dict["mig"])}')
            print(f'\tMIR: {np.mean(scores_dict["mir"])}')

    print(f'--Unsupervised--')
    if not in_HPO:
        print(f'\tGaussian TC: {np.mean(scores_dict["GTC"]):.4f}')
        print(f'\tGaussian WC: {np.mean(scores_dict["GWC"]):.4f}')
    print(f'\tMutual Info Score: {np.mean(scores_dict["MIS"]):.4f}')
    
    print('****************************************')
    print('****************************************')
    print('****************************************')

    print(f'[INFO] Computed with model {model_path}')
    print(f'[INFO] Computed with dataset {data_name}')

    if num_trials > 1 and SAVE_RESULTS:
        results = np.zeros((2,7))

        results[0,0] = np.mean(scores_dict["factorvae"])
        results[0,1] = np.mean(scores_dict["zmax"])
        results[0,2] = np.mean(scores_dict["mig"])
        results[0,3] = np.mean(scores_dict["mir"])
        results[0,4] = np.mean(scores_dict["GTC"])
        results[0,5] = np.mean(scores_dict["GWC"])
        results[0,6] = np.mean(scores_dict["MIS"])
        # print(f'MIS: {scores_dict["MIS"]}')
        
        results[1,0] = np.std(scores_dict["factorvae"])
        results[1,1] = np.std(scores_dict["zmax"])
        results[1,2] = np.std(scores_dict["mig"])
        results[1,3] = np.std(scores_dict["mir"])
        results[1,4] = np.std(scores_dict["GTC"])
        results[1,5] = np.std(scores_dict["GWC"])
        results[1,6] = np.std(scores_dict["MIS"])

        if PARAM_model_name != "": # Not initialized
            exec_name = PARAM_model_name
        else:
            exec_name = model_name.split("_")[2]
        save_path = f"disentanglement_lib/evaluation/results/{exec_name}_{data_name}_{num_trials}"
        print(f'\n[INFO] Saving results in {save_path}')
        np.save(save_path, results)




if __name__ == "__main__":
  #* <name> <data> <trials>
  if(len(sys.argv)) > 1:
        if (len(sys.argv) != 4):
            print(f'Usage: $python test.py <name> <data> <trials>')
            exit()
        PARAM_model_name = sys.argv[1]
        PARAM_data_name = sys.argv[2]
        PARAM_trials = int(sys.argv[3])
        SAVE_RESULTS = False # We don't want to store individual results while HPO
        in_HPO = True
        print(f'PARAM_model_name: {PARAM_model_name}, PARAM_data_name: {PARAM_data_name}, PARAM_trials: {PARAM_trials}')
        sys.argv = [sys.argv[0]] # Remove the remaining arguments so the program doesn't get mad
  
  absltest.main()
