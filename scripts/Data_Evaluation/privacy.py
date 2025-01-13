import matplotlib.pyplot as plt
from Data_Evaluation.membership_inference import evaluate_membership_attack
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
import gower
import pandas as pd
import numpy as np



def dcr(df_original, df_synth, model_name, dataset_nr, save_hist=False):
    """ Compute distance to closest record (DCR) and return the average distance.

    Parameters
    ----------
    df_original : pandas.DataFrame
        path to the original data
    df_synth : pandas.DataFrame
        path to the synthetic data
    model_name : str
        name of the model
    save_hist : bool, default=False
        save the histogram of distances

    Returns
    -------
    float
        average distance to the closest record
    """

    X = np.asarray(df_original, dtype=np.float64)
    Y = np.asarray(df_synth, dtype=np.float64)

    # Calculate Gower distance matrix
    distances = gower.gower_matrix(X, Y)

    if save_hist:
        # Plot histogram of distances
        plt.hist(distances.flatten(), bins=20, alpha=0.8)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Gower distances - {model_name}')
        plt.savefig(f'Evaluation_Results/Plots/DCR/Dataset_{dataset_nr}/dcr_hist_' + model_name + '.png' )

        # Clear the plot
        plt.clf()

    return np.mean(distances)

def nndr(df_original, df_synth):
    """ Compute nearest neighbor distance ratio (NNDR) and return the average ratio.

    Parameters
    ----------
    df_original : pandas.DataFrame
        path to the original data
    df_synth : pandas.DataFrame
        path to the synthetic data
        
    Returns
    -------
    float
        average nearest neighbor distance ratio
    """

    X = np.asarray(df_original, dtype=np.float64)
    Y = np.asarray(df_synth, dtype=np.float64)

    # Calculate Gower distance matrix
    distances = gower.gower_matrix(Y, X)

    # For each synthetic instance, find the two nearest neighbors in the original dataset
    nndr_values = []
    for row in distances:
        # Sort distances to find the nearest and second nearest neighbors
        sorted_distances = np.sort(row)
        if len(sorted_distances) > 1:  # Ensure there are at least two neighbors
            nndr = sorted_distances[0] / sorted_distances[1]
            # Set nndr to 0 if it is NaN
            if np.isnan(nndr):
                nndr = 0
            nndr_values.append(nndr)
        else:
            print('Warning: Not enough neighbors found for a synthetic instance')

    # Return the mean NNDR value
    return np.mean(nndr_values) if nndr_values else float('nan')


def mia(df_original, df_synth, model_name, dataset_nr, save_plts=False):
    """Perform membership inference attack and return the precision and accuracy values for different parameters.

    Parameters
    ----------
    df_original : pandas.DataFrame
        path to the original data
    df_synth : pandas.DataFrame
        path to the synthetic data
    save_plts : bool, default=False
        whether to save the accuracy and precision plots

    Returns
    -------
    dict 
        precision values for different thresholds
    dict 
        accuracy values for different thresholds
    """

    proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4]
    synth_indices = df_synth.index.tolist()

    precision_values = dict()
    accuracy_values = dict()

    for th in thresholds:
        precision_values[th] = []
        accuracy_values[th] = []

        for prop in proportions:
            attacker_data = df_original.sample(frac=prop, random_state=42)
            precision_val, accuracy_val = evaluate_membership_attack(attacker_data, synth_indices, df_synth, th)

            precision_values[th].append(round(precision_val, 2))
            accuracy_values[th].append(round(accuracy_val, 2))

    if save_plts:
        markers = ['v', '^', '<', '>']
        # Plot precision values
        plt.figure(figsize=(8, 6))
        for i, th in enumerate(thresholds):
            plt.plot(proportions, precision_values[th], label=f'Threshold: {th}', marker=markers[i])
        plt.xlabel('Proportions')
        plt.ylabel('Precision')
        plt.title(f'{model_name}: Precision')
        plt.legend()
        plt.savefig(f'Evaluation_Results/Plots/MIA/Dataset_{dataset_nr}/{model_name}_mia_precision.png')

        # Plot accuracy values
        plt.figure(figsize=(8, 6))
        for i, th in enumerate(thresholds):
            plt.plot(proportions, accuracy_values[th], label=f'Threshold: {th}', marker=markers[i])
        plt.xlabel('Proportions')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name}: Accuracy')
        plt.legend()
        plt.savefig(f'Evaluation_Results/Plots/MIA/Dataset_{dataset_nr}/{model_name}_mia_accuracy.png')
    
    result = {}
    result['precision'] = precision_values
    result['accuracy'] = accuracy_values
    
    return result
