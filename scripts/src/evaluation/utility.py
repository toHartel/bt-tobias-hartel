from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def run_utility_eval(df_original_train, df_original_test, df_synthetic, target_column, utility_function):
    """ Run selected utility model on the original and synthetic data and calculate the difference in accuracy and F1 score.

    Parameters
    ----------
    data_train : pandas.DataFrame
        Path to training data
    data_test : pandas.DataFrame
        Path to test data
    synth_data : pandas.DataFrame
        Path to synthetic data
    target_column : str
        Target variable used for prediction
    utility_function : str
        Utility function to evaluate the data. One of ["random_forest", "logistic_regression", "multilayer_perceptron"]

    Returns
    -------
    dict :
        Returns a dict with the following keys: "acc_original", "f1_original", "roc_auc_original", "acc_synth", "f1_synth", "roc_auc_synth", "acc_diff", "f1_diff", "roc_auc_diff"
    """
    model = ""
    if utility_function == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif utility_function == "logistic_regression":
        model = LogisticRegression(random_state=42, max_iter=1000, verbose=0)
    elif utility_function == "multilayer_perceptron":
        model = MLPClassifier(random_state=42)
    else:
        raise ValueError("Utility function not supported.")

    result = {}
    result["acc_original"], result["f1_original"], result["roc_auc_original"] = fit_model(df_original_train, df_original_test, target_column, model)
    result["acc_synth"], result["f1_synth"], result["roc_auc_synth"] = fit_model(df_synthetic, df_original_test, target_column, model)
    result["acc_diff"] = round(result["acc_original"] - result["acc_synth"], 2)
    result["f1_diff"] = round(result["f1_original"] - result["f1_synth"], 2)
    result["roc_auc_diff"] = round(result["roc_auc_original"] - result["roc_auc_synth"], 2)

    return result


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
    y_proba = model.predict_proba(X_test)

    accuracy_original, f1_score_original, roc_auc_original = calculate_metrics(y_test, y_pred, y_proba)

    return accuracy_original, f1_score_original, roc_auc_original

def calculate_metrics(y_test, y_pred, y_proba):
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
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    f1_score = round(classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score'], 2)
    if y_proba.shape[1] == 2:
        roc_auc = round(roc_auc_score(y_test, y_proba[:, 1]), 2)
    else:
        roc_auc = round(roc_auc_score(y_test, y_proba, multi_class='ovr'), 2)
    return accuracy, f1_score, roc_auc
