from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from dython.nominal import associations
import pandas as pd
import numpy as np

def pairwise_correlation_diff(original_path, synthetic_path):
    """
    Calculate the difference in pairwise correlation between the original and synthetic data.

    Parameters
    ----------
    original_path : str
        Path to the original data.
    synthetic_path : str
        Path to the synthetic data.

    Returns
    -------
    float
        returns the average absolute difference in pairwise correlation.
    """
    original_data = pd.read_csv(original_path)
    synthetic_data = pd.read_csv(synthetic_path)
    assoc_original = associations(original_data, plot=False, nom_nom_assoc='theil')
    assoc_synth = associations(synthetic_data, plot=False, nom_nom_assoc='theil')

    corr_matrix_original = assoc_original['corr']
    corr_matrix_synth = assoc_synth['corr']

    # compute the difference between the correlation matrices
    corr_diff = corr_matrix_original - corr_matrix_synth
    return corr_diff.abs().mean().mean()

def jsd(original_path, synthetic_path):
    """Calculate the average Jensen-Shannon divergence between the original and synthetic data.

    Parameters
    ----------
    original_path : str
        Path to the original data.
    synthetic_path : str
        Path to the synthetic data.

    Returns
    -------
    float
        Returns the average Jensen-Shannon divergence.
    """
    # JSD for categorical data ?
    df_original = pd.read_csv(original_path)
    df_synthetic = pd.read_csv(synthetic_path)

    # calculate Jensen-Shannon divergence for each column
    results = {}
    for col in df_original.columns:

        # calculate probability distributions
        freq_original = df_original[col].value_counts(normalize=True)
        freq_synth = df_synthetic[col].value_counts(normalize=True)
        
        # Reindex to make sure both distributions have the same categories
        categories = set(freq_original.index).union(set(freq_synth.index))
        freq_original = freq_original.reindex(categories, fill_value=0)
        freq_synth = freq_synth.reindex(categories, fill_value=0)
        
        # Calculate Jensen-Shannon distance
        js_distance = jensenshannon(freq_original, freq_synth)
        results[col] = js_distance ** 2 # square the JS distance to get the Jensen-Shannon divergence
        print(f"JSD for column {col}: {results[col]}")

    # Return average JSD
    return np.mean(list(results.values()))


def wd(original_path, synthetic_path):
    """Calculate the average Wasserstein distance between the original and synthetic data.

    Parameters
    ----------
    original_path : str
        Path to the original data.
    synthetic_path : str
        Path to the synthetic data.

    Returns
    -------
    float
        Returns the average Wasserstein distance.
    """
    # WD for continuous data?x
    df_original = pd.read_csv(original_path)
    df_synthetic = pd.read_csv(synthetic_path)

    results = {}
    for col in df_original.columns:

        # calculate probability distributions
        freq_original = df_original[col].value_counts(normalize=True)
        freq_synth = df_synthetic[col].value_counts(normalize=True)

        # Calculate Wasserstein distance for each column
        results[col] = wasserstein_distance(freq_original, freq_synth)
        print(f"WD for column {col}: {results[col]}")

    # Return average WD
    return np.mean(list(results.values()))