
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum


def ILP(data, weights, lambda_weights, gamma_weights, user_income, affordability_ratio):
    w1, w2, w3, w4, w5, w6 = weights

    data['AirQualityScore'] = (
        lambda_weights[0] * data['Good_Percentage'] +
        lambda_weights[1] * data['Moderate_Percentage'] +
        lambda_weights[2] * data['Unhealthy_Sensitive_Percentage'] +
        lambda_weights[3] * data['Unhealthy_Percentage'] +
        lambda_weights[4] * data['Very_Unhealthy_Percentage'] +
        lambda_weights[5] * data['Hazardous_Percentage']
    )

    data['CrimeScore'] = (
        gamma_weights[0] * data['Arson Count'] +
        gamma_weights[1] * data['Property Crimes Count'] +
        gamma_weights[2] * data['Violent Crimes Count']
    )

    model = Model("Metro_Selection")
    x = model.addVars(data.index, vtype=GRB.BINARY, name="x")

    objective = quicksum(
        (w1 * data.loc[i, 'Predicted_HomeValue'] / data.loc[i, 'Hourly_Wage']
         - w2 * data.loc[i, 'HealthCareFacilityAmmount']
         - w3 * data.loc[i, 'AirQualityScore']
         + w4 * data.loc[i, 'Unemployment Rate']
         + w5 * data.loc[i, 'CrimeScore']
         + w6 * data.loc[i, 'Population Per Square Mile (Land Area)']) * x[i]
        for i in data.index
    )
    model.setObjective(objective, GRB.MINIMIZE)

    for i in data.index:
        model.addConstr(
            x[i] * data.loc[i, 'Predicted_HomeValue'] <= x[i] * affordability_ratio * user_income * 12 * 40,
            name=f"affordability_{i}"
        )

    model.addConstr(quicksum(x[i] for i in data.index) == 1, name="select_one_metro")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        for i in data.index:
            if x[i].x > 0.5:
                return {
                    "Method": "ILP",
                    "Selected Metro": data.loc[i, 'Metro'],
                    "Objective Value": model.objVal
                }
    return {
        "Method": "ILP",
        "Selected Metro": "No solution found",
        "Objective Value": np.nan
    }

def compute_objective(data, i, weights, lambda_weights, gamma_weights, user_income, affordability_ratio):
    w1, w2, w3, w4, w5, w6 = weights

    affordability_penalty = 1e6 if data.loc[i, 'Predicted_HomeValue'] > affordability_ratio * user_income * 12 * 40 else 0

    return (
        w1 * data.loc[i, 'Predicted_HomeValue'] / data.loc[i, 'Hourly_Wage']
        - w2 * data.loc[i, 'HealthCareFacilityAmmount']
        - w3 * (
            lambda_weights[0] * data.loc[i, 'Good_Percentage'] +
            lambda_weights[1] * data.loc[i, 'Moderate_Percentage'] +
            lambda_weights[2] * data.loc[i, 'Unhealthy_Sensitive_Percentage'] +
            lambda_weights[3] * data.loc[i, 'Unhealthy_Percentage'] +
            lambda_weights[4] * data.loc[i, 'Very_Unhealthy_Percentage'] +
            lambda_weights[5] * data.loc[i, 'Hazardous_Percentage']
        )
        + w4 * data.loc[i, 'Unemployment Rate']
        + w5 * (
            gamma_weights[0] * data.loc[i, 'Arson Count'] +
            gamma_weights[1] * data.loc[i, 'Property Crimes Count'] +
            gamma_weights[2] * data.loc[i, 'Violent Crimes Count']
        )
        + w6 * data.loc[i, 'Population Per Square Mile (Land Area)']
        + affordability_penalty
    )

def SA(data, weights, lambda_weights, gamma_weights, user_income, affordability_ratio, max_iter=1000, temp=1000, cooling=0.95):
    current = np.random.choice(data.index)
    best = current
    best_score = compute_objective(data, current, weights, lambda_weights, gamma_weights, user_income, affordability_ratio)

    for _ in range(max_iter):
        neighbor = np.random.choice(data.index)
        delta = compute_objective(data, neighbor, weights, lambda_weights, gamma_weights, user_income, affordability_ratio) - best_score

        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current = neighbor
            best_score += delta
            best = current

        temp *= cooling
        if temp < 1e-3:
            break

    if best_score > 1e5:
        return {"Method": "SA", "Selected Metro": "No solution", "Objective Value": np.nan}

    return {
        "Method": "SA",
        "Selected Metro": data.loc[best, 'Metro'],
        "Objective Value": best_score
    }