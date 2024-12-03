from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd


    

def run_utility_eval(data_train, data_test, synth_data, target_column, utility_function):
    """ Run selected utility model on the original and synthetic data and calculate the difference in accuracy and F1 score.
    
    Parameters
    ----------
    data_train : str
        Path to training data
    data_test : str
        Path to test data
    synth_data : str
        Path to synthetic data
    target_column : str
        Target variable used for prediction
    utility_function : str
        Utility function to evaluate the data. One of ["random_forest", "logistic_regression", "multilayer_perceptron"]

    Returns
    -------
    float, float :
        Returns the absolute difference in accuracy and F1 score between the original and synthetic data.
    """
    model = ""
    if utility_function == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif utility_function == "logistic_regression":
        model = LogisticRegression(random_state=42)
    elif utility_function == "multilayer_perceptron":
        model = MLPClassifier(random_state=42)
    else:
        raise ValueError("Utility function not supported.")

    df_original_train = pd.read_csv(data_train)
    df_original_test = pd.read_csv(data_test)
    df_synthetic = pd.read_csv(synth_data)

    # If the dataset contains categorical columns, convert them to one-hot encoding
    if df_original_train.select_dtypes(include=['object']).shape[1] > 0:
        # Combine all DataFrames to ensure the same columns
        combined_df = pd.concat([df_original_train, df_original_test, df_synthetic], axis=0)
        combined_df = pd.get_dummies(combined_df)

        # Split the DataFrames again
        df_original_train = combined_df.iloc[:len(df_original_train), :]
        df_original_test = combined_df.iloc[len(df_original_train):len(df_original_train) + len(df_original_test), :]
        df_synthetic = combined_df.iloc[len(df_original_train) + len(df_original_test):, :]

    # Reindex the test data to match the training data
    #df_original_test = df_original_test.reindex(columns=df_original_train.columns, fill_value=0)

    accuracy_original, f1_score_original = fit_model(df_original_train, df_original_test, target_column, model)
    accuracy_synth, f1_score_synth = fit_model(df_synthetic, df_original_test, target_column, model)

    accuracy_diff = abs(accuracy_original - accuracy_synth)
    f1_score_diff = abs(f1_score_original - f1_score_synth)

    return accuracy_diff, f1_score_diff


def fit_model(df_train, df_test, target_column, model):
    """ Fit selected model on the training data and calculate the accuracy and F1 score on the test data.

    Parameters
    ----------
    df_train : pandas.DataFrame
        Training data
    df_test : pandas.DataFrame
        Test data
    target_column : str
        Target variable used for prediction
    model : sklearn.base.BaseEstimator
        A Scikit-Learn estimator. One of [RandomForestClassifier, LogisticRegression, MLPClassifier]

    Returns
    -------
    float, float 
        Returns the accuracy and F1 score of the predictions.
    """
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy_original, f1_score_original = calculate_metrics(y_test, y_pred)

    return accuracy_original, f1_score_original

def calculate_metrics(y_test, y_pred):
    """ Calculate the accuracy and F1 score of the predictions.

    Parameters
    ----------
    y_test : 
        Target variables from the test data
    y_pred :
        Predictions made by the model

    Returns
    -------
    float, float 
        Returns the accuracy and F1 score of the predictions.
    """
    accuracy = accuracy_score(y_test, y_pred)
    f1_score = classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']

    return accuracy, f1_score
