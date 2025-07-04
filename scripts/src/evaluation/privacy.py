import matplotlib.pyplot as plt
from evaluation.membership_inference import evaluate_membership_attack
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.spatial.distance import cdist

def dcr(df_original, df_synth, model_name, dataset_name, within=False, save_hist=False):
    """ Compute distance to closest record (DCR) and return the 5th percentile.

    Parameters
    ----------
    df_original : pandas.DataFrame
        Dataframe of the original data
    df_synth : pandas.DataFrame
        Dataframe of the synthetic data
    model_name : str
        name of the model
    dataset_name : str
        name of the original dataset
    within : {"Original",  "Synthetic", False}, default=False
        whether to compute dcr within the original or synthetic data or between the two
    save_hist : bool, default=False
        save the histogram of distances

    Returns
    -------
    float
        average distance to the closest record
    """

    X = np.asarray(df_original, dtype=np.float64)
    Y = np.asarray(df_synth, dtype=np.float64)

    if (within=="Original"):
        distances = cdist(X, X, 'euclidean')
        # Set the diagonal to infinity to exclude the distance to itself
        np.fill_diagonal(distances, np.inf)
    elif (within=="Synthetic"):
        distances = cdist(Y, Y, 'euclidean')
        np.fill_diagonal(distances, np.inf)
    else:
        distances = cdist(Y, X, 'euclidean')
   
    # Find the minimum distance for each synthetic instance (DCR)
    dcr_values = np.min(distances, axis=1)

    if save_hist:
        # Plot histogram of distances
        bins = np.arange(0, 50, 1)
        dcr_values_clipped = np.clip(dcr_values, bins[0], bins[-1])
        plt.hist(dcr_values_clipped.flatten(), bins=bins, alpha=0.8)
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title(f'DCR values - {model_name}')
        plt.savefig(f'../data/results/plots/dcr/{dataset_name}/{model_name}_dcr_hist.png')

        # Clear the plot
        plt.clf()

    # return np.mean(dcr_values)
    return round(np.percentile(dcr_values, 5), 3)

def nndr(df_original, df_synth, within=False):
    """ Compute nearest neighbor distance ratio (NNDR) and return the 5th percentile.

    Parameters
    ----------
    df_original : pandas.DataFrame
        Dataframe of the original data
    df_synth : pandas.DataFrame
        Dataframe of the synthetic data
    within : {"Original",  "Synthetic", False}, default=False
        whether to compute nndr within the original or synthetic data or between the two
        
    Returns
    -------
    float
        average nearest neighbor distance ratio
    """

    X = np.asarray(df_original, dtype=np.float64)
    Y = np.asarray(df_synth, dtype=np.float64)

    # Calculate distance matrix
    if (within=="Original"):
        distances = cdist(X, X, 'euclidean')
        # Set the diagonal to infinity to exclude the distance to itself
        np.fill_diagonal(distances, np.inf)
    elif (within=="Synthetic"):
        distances = cdist(Y, Y, 'euclidean')
        np.fill_diagonal(distances, np.inf)
    else:
        distances = cdist(Y, X, 'euclidean')


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
    return round(np.percentile(nndr_values, 5), 3)

def mia(df_original, df_train, df_synth, model_name, dataset_name, save_plts=False):
    """Perform membership inference attack and return the precision and accuracy values for different parameters.

    Parameters
    ----------
    df_orignal : pandas.DataFrame
        Dataframe of the original data
    df_train : pandas.DataFrame
        Dataframe of the training data
    df_synth : pandas.DataFrame
       Dataframe of the synthetic data
    model_name : str
        name of the model
    dataset_name : str
        name of the original dataset
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
    train_data_indices = df_train.index.tolist()

    precision_values = dict()
    accuracy_values = dict()

    for th in thresholds:
        precision_values[th] = []
        accuracy_values[th] = []

        for prop in proportions:
            attacker_data = df_original.sample(frac=prop, random_state=42)
            precision_val, accuracy_val = evaluate_membership_attack(attacker_data, train_data_indices, df_synth, th)

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
        plt.savefig(f'../data/results/plots/mia/{dataset_name}/{model_name}_mia_precision.png')

        # Plot accuracy values
        plt.figure(figsize=(8, 6))
        for i, th in enumerate(thresholds):
            plt.plot(proportions, accuracy_values[th], label=f'Threshold: {th}', marker=markers[i])
        plt.xlabel('Proportions')
        plt.ylabel('Accuracy')
        plt.title(f'{model_name}: Accuracy')
        plt.legend()
        plt.savefig(f'../data/results/plots/mia/{dataset_name}/{model_name}_mia_accuracy.png')
    
    result = {}
    result['precision'] = precision_values
    result['accuracy'] = accuracy_values
    
    return result
