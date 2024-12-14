# Author: Yihan Zhou
# Date: 2024-12-14

import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


class LinearModel:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
class RandomForestModel:
    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
class GradientBoostingModel:
    def __init__(self):
        self.model = GradientBoostingRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
def train_model(model, X, y):
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return mean_squared_error(y, y_pred)

def save_model(model, model_dir, model_name):
    # default model directory: ./models

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_name)
    print(f"Saving model to {model_path}")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path


# example usage
if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    model = LinearModel()
    model = train_model(model, X, y)
    print(evaluate_model(model, X, y))

