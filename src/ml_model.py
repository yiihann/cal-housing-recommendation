
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(X, y, model_path="./models/model.pkl"):
    """
    Train a Random Forest model and save it.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        model_path (str): Path to save the trained model.
    Returns:
        model: Trained model.
    """
    # Standardize features and target
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    joblib.dump(feature_scaler, "./models/feature_scaler.pkl")

    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()
    joblib.dump(target_scaler, "./models/target_scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    joblib.dump(model, model_path)
    return model

def load_model(model_path="./models/model.pkl"):
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
        np.ndarray: Predicted values in original scale.
    """
    feature_scaler = joblib.load("../models/feature_scaler.pkl")
    target_scaler = joblib.load("../models/target_scaler.pkl")
    X_scaled = feature_scaler.transform(X)
    y_scaled_pred = model.predict(X_scaled)
    return target_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1)).ravel()
