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

def sdv_generate_data(data_path: str, num_samples: int = 1000) -> None:

    data = pd.read_csv(data_path)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    copula_gan = CopulaGANSynthesizer(metadata)
    ctgan = CTGANSynthesizer(metadata)
    tvae = TVAESynthesizer(metadata)
    gaussian_copula = GaussianCopulaSynthesizer(metadata)
    
    # Fit und Sample für copula_gan
    start_time = time.time()
    print("Fitting copula_gan...\n")
    copula_gan.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit copula_gan: {fit_time:.2f} seconds\n")
    copula_gan_samples = copula_gan.sample(num_samples)
    copula_gan_samples.to_csv('Synthetic_Data/copula_gan_samples.csv', sep = ",", index = False)


    # Fit und Sample für ctgan
    start_time = time.time()
    print("Fitting ctgan...\n")
    ctgan.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit ctgan: {fit_time:.2f} seconds\n")
    ctgan_samples = ctgan.sample(num_samples)
    ctgan_samples.to_csv('Synthetic_Data/ctgan_samples.csv', sep = ",", index = False)


    # Fit und Sample für tvae
    start_time = time.time()
    print("Fitting tvae...\n")
    tvae.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit tvae: {fit_time:.2f} seconds\n")
    tvae_samples = tvae.sample(num_samples)
    tvae_samples.to_csv('Synthetic_Data/tvae_samples.csv', sep = ",", index = False)


    # Fit und Sample für gaussian_copula
    start_time = time.time()
    print("Fitting gaussian_copula...\n")
    gaussian_copula.fit(data)
    fit_time = time.time() - start_time
    print(f"Time taken to fit gaussian_copula: {fit_time:.2f} seconds\n")
    gaussian_samples = gaussian_copula.sample(num_samples)
    gaussian_samples.to_csv('Synthetic_Data/gaussian_samples.csv', sep = ",", index = False)
