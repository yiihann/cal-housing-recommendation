
import argparse
import pandas as pd
from src.data_processing import preprocess_pipeline, prepare_ml_dataset, prepare_optimization_dataset
from src.ml_model import train_model, load_model, predict
from src.optimization import ILP, SA
def main(target_column, income, weights, train=False, model_path="./models/model.pkl"):
    """
    Main function to run the pipeline: data processing, ML prediction, and optimization.
    Args:
        target_column (str): Name of the target variable for ML model.
        income (float): User's income for optimization.
        weights (dict): Weights for optimization factors.
        train (bool): Whether to train a new ML model or load an existing one.
        model_path (str): Path to save or load the ML model.
    """
    print("Running full preprocessing pipeline...")
    df_full = preprocess_pipeline()
    df_accessed = df_full[df_full['Year']<2023]
    df_held_out = df_full[df_full['Year']>=2023]

    print("Preparing data for ML model...")
    df_ml = prepare_ml_dataset(df_accessed)
    df_ml.to_csv("./data/processed/ml_data.csv", index=False)
    X, y = df_ml.drop(columns=[target_column]), df_ml[target_column]
    print("Training or loading model...")
    if train:
        model = train_model(X, y, model_path)
    else:
        model = load_model(model_path)

    # print("Predicting on holdout data...")
    # X_held_out, y_held_out = df_held_out.drop(columns=[target_column]), df_held_out[target_column]
    # y_pred = predict(model, X_held_out)
    
    print("Preparing optimization data...")
    hourly_wage = pd.read_excel("./data/raw/wage_housing.xlsx", sheet_name='Data')
    df_opt = prepare_optimization_dataset(df_full, hourly_wage)
    df_opt.to_csv("./data/processed/optimization_data.csv", index=False)

    # Define optimization parameters
    affordability_ratio = 0.4
    lambda_weights = [0.6, 0.4, -0.6, -0.8, -1.0, -1.2]
    gamma_weights = [1, 0.5, 0.7]

    print("\nRunning ILP Optimization...")
    ilp_result = ILP(df_opt[df_opt["Year"] == 2024].reset_index(drop=True),
                     weights.values(), lambda_weights, gamma_weights, income, affordability_ratio)
    print(ilp_result)

    print("\nRunning Simulated Annealing Optimization...")
    sa_result = SA(df_opt[df_opt["Year"] == 2024].reset_index(drop=True),
                   weights.values(), lambda_weights, gamma_weights, income, affordability_ratio)
    print(sa_result)

    # print("\nTop Recommended Locations:")
    # print(recommendations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="California Housing Optimization Pipeline")
    parser.add_argument("--target_column", type=str, default="HomeValue", help="Target column for ML model")
    parser.add_argument("--income", type=float, required=True, help="User's income for optimization")
    parser.add_argument("--weights", type=dict, default={
        "affordability": 0.4,
        "healthcare": 0.2,
        "air_quality": 0.15,
        "unemployment": 0.1,
        "crime": 0.1,
        "density": 0.05
    }, help="Weights for optimization factors")
    parser.add_argument("--train", action="store_true", help="Train a new model instead of loading")
    parser.add_argument("--model_path", type=str, default="model.pkl", help="Path to save or load ML model")
    args = parser.parse_args()

    main(args.target_column, args.income, args.weights, args.train, args.model_path)
