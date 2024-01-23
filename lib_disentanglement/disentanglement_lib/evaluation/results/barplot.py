"""
    Script to save a barplot containing the mean and std of the different metrics
    used to evaluate our models (one barplot per metric).

    Based in https://pythonforundergradengineers.com/python-matplotlib-error-bars.html
"""


import numpy as np
import matplotlib.pyplot as plt

METRICS_NAMES = ["FactorVAE ↑", "Z-Max ↑", "MIG ↑", "MIR ↑", "GTC ↓", "GWC ↓", "MIS ↓"]

# ! Load here the results to use
metrics_easy = np.load("test62_test-easy_15.npy")
metrics_mat_ill = np.load("test62_test-mat_ill_15.npy")
metrics_mat = np.load("test62_test-materials_15.npy")
metrics_ill = np.load("test62_test-illuminations_15.npy")
metrics_analytic = np.load("test62_test-analytic_15.npy")


for metric_idx in range(len(METRICS_NAMES)):
    # Create lists for the plot
    models = ['mat+ill+analytic', 'mat+ill', 'mat', 'ill', 'analytic'] #* Names for the X axis
    x_pos = np.arange(len(models))
    # ! Use here the loaded results
    means = np.array([metrics_easy[0,metric_idx], metrics_mat_ill[0,metric_idx], metrics_mat[0,metric_idx], metrics_ill[0,metric_idx], metrics_analytic[0,metric_idx]])
    stds = np.array([metrics_easy[1,metric_idx], metrics_mat_ill[1,metric_idx], metrics_mat[1,metric_idx], metrics_ill[1,metric_idx], metrics_analytic[1,metric_idx]])
    stds = 2 * stds

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(f'{METRICS_NAMES[metric_idx]}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models)
    # ax.set_title(f'Comparison of {METRICS_NAMES[0]} score')
    ax.yaxis.grid(True)

    if metric_idx > 3:
        plt.yscale("log")


    # Save the figure and show
    plt.tight_layout()
    plt.savefig(f'Results_{METRICS_NAMES[metric_idx]}.png')
    plt.show()
