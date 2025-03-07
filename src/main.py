import argparse
import pandas as pd
from data_processing import preprocess_pipeline
from ml_model import train_model, load_model, predict
from optimization import setup_optimization_problem, run_optimization, postprocess_results

def main(data_path, target_column, income, weights, train=False, model_path="model.pkl"):
    """
    Main function to run the pipeline: data processing, ML prediction, and optimization.
    Args:
        data_path (str): Path to the dataset file.
        target_column (str): Name of the target variable for ML model.
        income (float): User's income for optimization.
        weights (dict): Weights for optimization factors.
        train (bool): Whether to train a new ML model or load an existing one.
        model_path (str): Path to save or load the ML model.
    """
    print("Processing data...")
    data = preprocess_pipeline(data_path)

    print("Loading or training ML model...")
    X, y = data.drop(columns=[target_column]), data[target_column]
    
    if train:
        model = train_model(X, y, model_path)
    else:
        model = load_model(model_path)

    print("Making predictions...")
    data["PredictedPrice"] = predict(model, X)

    print("Setting up optimization problem...")
    problem_setup = setup_optimization_problem(data, income, weights)

    print("Running optimization...")
    solution = run_optimization(problem_setup)

    print("Processing results...")
    recommendations = postprocess_results(data, solution)

    print("\nTop Recommended Locations:")
    print(recommendations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="California Housing Optimization Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--target_column", type=str, default="HomeValue", help="Target column for ML model")
    parser.add_argument("--income", type=float, required=True, help="User's income for optimization")
    parser.add_argument("--weights", type=dict, default={"affordability": 0.4, "healthcare": 0.2, "air_quality": 0.15, "crime": 0.1, "unemployment": 0.1}, help="Weights for optimization factors")
    parser.add_argument("--train", action="store_true", help="Train a new model instead of loading")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path to save or load ML model")

    args = parser.parse_args()
    main(args.data_path, args.target_column, args.income, args.weights, args.train, args.model_path)
