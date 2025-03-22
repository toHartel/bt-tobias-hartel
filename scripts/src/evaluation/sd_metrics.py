import pandas as pd
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality
import os

def sd_metrics(train_data, synthetic_data) -> None:
    """
    Evaluate the quality of the synthetic data using SDMetrics from SDV.
    
    Parameters
    ----------
    train_data: pandas.DataFrame 
        DataFrame of original training data 
    synthetic_data: str 
        DataFrame of synthetic data 

    Returns
    -------
    pandas.DataFrame
        Returns the diagnostic and quality metrics of the synthetic data.
    """

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)

    diagnostic = run_diagnostic(train_data, synthetic_data, metadata, verbose=False)
    quality = evaluate_quality(train_data, synthetic_data, metadata, verbose=False)

    diagnostic_df = diagnostic.get_properties().set_index('Property').T
    quality_df = quality.get_properties().set_index('Property').T
    
    combined_df = pd.concat([diagnostic_df, quality_df], axis=1)
    
    return combined_df