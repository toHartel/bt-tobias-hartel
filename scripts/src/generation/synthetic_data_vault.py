import pandas as pd
import sdv
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CopulaGANSynthesizer
import time

def sdv_generate_data(
    data_path: str, 
    dataset_name: str, 
    num_samples: int = 1000
) -> None:
    """ Generate synthetic data using the models CopulaGAN, CTGAN, TVAE and GaussianCopula from the SDV library

    Parameters
    ----------
    data_path: str 
        The path to the input data file
    dataset_nr: str 
        The number of the dataset
    num_samples: int, defualt = 1000
        Number of samples to generate

    Returns
    -------
    None

    """
    # Load dataset
    data = pd.read_csv(data_path)

    # Detect metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    # Initialize the synthesizers
    copula_gan = CopulaGANSynthesizer(metadata)
    ctgan = CTGANSynthesizer(metadata)
    tvae = TVAESynthesizer(metadata, enforce_min_max_values=True)
    gaussian_copula = GaussianCopulaSynthesizer(metadata)
    
    # Fit and sample copula_gan
    start_time = time.time()
    print("Fitting copula_gan...\n")
    copula_gan.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit copula_gan: {fit_time:.2f} seconds\n")
    copula_gan_samples = copula_gan.sample(num_samples)
    copula_gan_samples.to_csv(f'../data/synthetic_data/{dataset_name}/copula_gan.csv', sep = ",", index = False)


    # Fit and sample ctgan
    start_time = time.time()
    print("Fitting ctgan...\n")
    ctgan.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit ctgan: {fit_time:.2f} seconds\n")
    ctgan_samples = ctgan.sample(num_samples)
    ctgan_samples.to_csv(f'../data/synthetic_data/{dataset_name}/ctgan.csv', sep = ",", index = False)


    # Fit and sample tvae
    start_time = time.time()
    print("Fitting tvae...\n")
    tvae.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit tvae: {fit_time:.2f} seconds\n")
    tvae_samples = tvae.sample(num_samples)
    tvae_samples.to_csv(f'../data/synthetic_data/{dataset_name}/tvae.csv', sep = ",", index = False)


    # Fit and sample gaussian_copula
    start_time = time.time()
    print("Fitting gaussian_copula...\n")
    gaussian_copula.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit gaussian_copula: {fit_time:.2f} seconds\n")
    gaussian_samples = gaussian_copula.sample(num_samples)
    gaussian_samples.to_csv(f'../data/synthetic_data/{dataset_name}/gaussian_copula.csv', sep = ",", index = False)
