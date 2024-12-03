import matplotlib.pyplot as plt
from Data_Evaluation.membership_inference import evaluate_membership_attack
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import NearestNeighbors
import gower
import pandas as pd
import numpy as np



def dcr(path_original, path_synth, model, save_hist=False):
    """ Compute distance to closest record (DCR) and return the average distance.

    Parameters
    ----------
    path_original : str
        path to the original data
    path_synth : str
        path to the synthetic data
    model : str
        name of the model
    save_hist : bool, default=False
        save the histogram of distances

    Returns
    -------
    float
        average distance to the closest record
    """
    data_original = pd.read_csv(path_original)
    data_synth = pd.read_csv(path_synth)
    X = np.asarray(data_original)
    Y = np.asarray(data_synth)

    # Calculate Gower distance matrix
    distances = gower.gower_matrix(X, Y)

    if save_hist:
        # Plot histogram of distances
        plt.hist(distances.flatten(), bins=20, alpha=0.8)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Gower distances - {model}')
        plt.savefig('Plots/dcr_hist_' + model + '.png' )

        # Clear the plot
        plt.clf()

    return np.mean(distances)

def nndr(path_original, path_synth):
    """ Compute nearest neighbor distance ratio (NNDR) and return the average ratio.

    Parameters
    ----------
    path_original : str
        path to the original data
    path_synth : str
        path to the synthetic data
        
    Returns
    -------
    float
        average nearest neighbor distance ratio
    """
    # Load datasets
    data_original = pd.read_csv(path_original)
    data_synth = pd.read_csv(path_synth)

    # Calculate Gower distance matrix
    distances = gower.gower_matrix(data_synth, data_original)

    # For each synthetic instance, find the two nearest neighbors in the original dataset
    nndr_values = []
    for row in distances:
        # Sort distances to find the nearest and second nearest neighbors
        sorted_distances = np.sort(row)
        if len(sorted_distances) > 1:  # Ensure there are at least two neighbors
            nndr = sorted_distances[0] / sorted_distances[1]
            nndr_values.append(nndr)

    # Return the mean NNDR value
    return np.mean(nndr_values) if nndr_values else float('nan')


def mia(path_original, path_synth, save_plts=False):
    """Perform membership inference attack and return the precision and accuracy values for different parameters.

    Parameters
    ----------
    path_original : str
        path to the original data
    path_synth : str
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
    # Load datasets
    data_original = pd.read_csv(path_original)
    data_synth = pd.read_csv(path_synth)


    proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4]
    synth_indices = data_synth.index.tolist()

    precision_values = dict()
    accuracy_values = dict()

    for th in thresholds:
        precision_values[th] = []
        accuracy_values[th] = []

        for prop in proportions:
            attacker_data = data_original.sample(frac=prop, random_state=42)
            precision_val, accuracy_val = evaluate_membership_attack(attacker_data, synth_indices, data_synth, th)

            precision_values[th].append(precision_val)
            accuracy_values[th].append(accuracy_val)
    
    if save_plts:
        # Plot precision values
        plt.figure(figsize=(8, 6))
        for th in thresholds:
            plt.plot(proportions, precision_values[th], label=f'Threshold: {th}', marker='o')
        plt.xlabel('Proportions')
        plt.ylabel('Precision')
        plt.title('Precision')
        plt.legend()
        plt.savefig('mia_precision.png')

        # Plot accuracy values
        plt.figure(figsize=(8, 6))
        for th in thresholds:
            plt.plot(proportions, accuracy_values[th], label=f'Threshold: {th}', marker='o')
        plt.xlabel('Proportions')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig('mia_accuracy.png')
    
    return precision_values, accuracy_values
