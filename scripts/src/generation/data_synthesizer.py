from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator

def ds_generate_data(
    data_path: str, 
    dataset_name: str,
    epsilon: float = 1.0, 
    degree_of_bayesian_network: int = 2, 
    num_samples: int = 1000, 
) -> None:

    """ Generate synthetic data using DataSynthesizer in correlated attribute mode.
    Parameters:
    ----------- 
    data_path: str
        The path to the input data file
    epsilon: float
        The privacy parameter
    degree_of_bayesian_network: int
        The degree of the Bayesian network
    num_samples: int
        The number of tuples to generate
    output_path: str
        The path to save the synthetic data
    dataset_name: str
        The name of the original dataset
    Returns:
    --------
    None
    """
    # Set mode that DataSynthesizer will run in
    mode = 'correlated_attribute_mode'
    # Set the path to the description file that will be generated
    description_file = f'description_{mode}.json'

    # Instantiate the DataDescriber and generate the description file from the dataset
    describer = DataDescriber()
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=data_path, epsilon=epsilon, k=degree_of_bayesian_network)
    describer.save_dataset_description_to_file(description_file)

    # Generate synthetic data using the description file
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_samples, description_file)
    output_path = f'../data/synthetic_data/{dataset_name}/ds.csv'
    generator.save_synthetic_data(output_path)



