

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(file_path, target_column):
    """
    Load dataset and split into features and target variable.
    Args:
        file_path (str): Path to the dataset file.
        target_column (str): Name of the target variable.
    Returns:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.
    """
    df = pd.read_csv(file_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def train_model(X, y, model_path="model.pkl"):
    """
    Train a Random Forest model and save it.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        model_path (str): Path to save the trained model.
    Returns:
        model: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Model Performance: RMSE={mean_squared_error(y_test, y_pred, squared=False)}, R2={r2_score(y_test, y_pred)}")

    joblib.dump(model, model_path)
    return model

def load_model(model_path="model.pkl"):
    """
    Load a trained model from file.
    Args:
        model_path (str): Path to the saved model file.
    Returns:
        model: Loaded model.
    """
    return joblib.load(model_path)

def predict(model, X):
    """
    Make predictions using the trained model.
    Args:
        model: Trained model.
        X (pd.DataFrame): Feature matrix.
    Returns:
        np.ndarray: Predicted values.
    """
    return model.predict(X)