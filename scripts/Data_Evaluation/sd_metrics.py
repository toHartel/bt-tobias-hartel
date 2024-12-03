import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import os

def sd_metrics(original_data: str, synthetic_data: str) -> None:
    """
    Evaluate the quality of the synthetic data using SDMetrics from SDV.
    
    Parameters
    ----------
    original_data: str 
        Path to the original data
    synthetic_data: str 
        Path to the synthetic data

    Returns
    -------
    pandas.DataFrame
        Returns the diagnostic and quality metrics of the synthetic data.
    """
    data = pd.read_csv(original_data)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    synthetic_data = pd.read_csv(synthetic_data)

    diagnostic = run_diagnostic(data, synthetic_data, metadata, verbose=False)
    quality = evaluate_quality(data, synthetic_data, metadata, verbose=False)

    diagnostic_df = diagnostic.get_properties().set_index('Property').T
    quality_df = quality.get_properties().set_index('Property').T
    
    combined_df = pd.concat([diagnostic_df, quality_df], axis=1)
    
    return combined_df