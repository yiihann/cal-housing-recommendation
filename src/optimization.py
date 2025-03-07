

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def setup_optimization_problem(data, income, weights):
    """
    Set up the optimization problem by defining the objective function and constraints.
    Args:
        data (pd.DataFrame): Processed dataset containing relevant metrics.
        income (float): User's income.
        weights (dict): Dictionary containing weights for affordability, healthcare, air quality, etc.
    Returns:
        dict: Contains the problem setup, including the objective function and constraints.
    """
    def quality_of_life(x):
        """
        Quality of Life function to maximize.
        Args:
            x (np.ndarray): Array of decision variables representing location scores.
        Returns:
            float: Negative QoL score (since we minimize in scipy.optimize).
        """
        affordability = np.dot(x, data["HourlyWageRequired"]) / income
        healthcare = np.dot(x, data["HealthcareScore"])
        air_quality = np.dot(x, data["AirQualityScore"])
        crime = np.dot(x, data["CrimeScore"])
        unemployment = np.dot(x, data["UnemploymentScore"])
        
        return -(weights["affordability"] * affordability + 
                 weights["healthcare"] * healthcare + 
                 weights["air_quality"] * air_quality - 
                 weights["crime"] * crime - 
                 weights["unemployment"] * unemployment)

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Ensures sum of selection probabilities is 1
        {"type": "ineq", "fun": lambda x: income - np.dot(x, data["HourlyWageRequired"])}  # Affordability constraint
    ]
    
    bounds = [(0, 1) for _ in range(len(data))]  # Each location's selection weight is between 0 and 1
    
    return {"objective": quality_of_life, "constraints": constraints, "bounds": bounds}

def run_optimization(problem_setup):
    """
    Run the optimization solver.
    Args:
        problem_setup (dict): Contains objective function, constraints, and bounds.
    Returns:
        np.ndarray: Optimized selection of locations.
    """
    num_locations = len(problem_setup["bounds"])
    initial_guess = np.ones(num_locations) / num_locations  # Equal weight initialization

    result = minimize(
        problem_setup["objective"],
        initial_guess,
        bounds=problem_setup["bounds"],
        constraints=problem_setup["constraints"],
        method="SLSQP"
    )

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)

def postprocess_results(data, solution):
    """
    Post-process optimization results and return top recommended locations.
    Args:
        data (pd.DataFrame): Processed dataset with location information.
        solution (np.ndarray): Optimized selection of locations.
    Returns:
        pd.DataFrame: Top recommended locations.
    """
    data["SelectionScore"] = solution
    return data.sort_values(by="SelectionScore", ascending=False).head(5)