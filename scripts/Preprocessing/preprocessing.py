import pandas as pd


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by removing missing values and duplicates and use one-hot-encoding.

    Parameters
    ----------
    data: pandas.DataFrame
        The input data to be preprocessed.

    Returns
    -------
    pandas.DataFrame
        The preprocessed data.
    """
    # Remove missing values
    data = data.dropna()
    # Remove duplicates
    data = data.drop_duplicates()

    print(data.tail())