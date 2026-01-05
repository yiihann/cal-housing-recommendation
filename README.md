# California Housing Affordability Optimization Project

## **Overview**
This project explores **housing affordability in California** by integrating **predictive modeling** and **optimization techniques**. The objective is to provide **data-driven insights** and **optimal housing recommendations** for policymakers, city planners, and individuals.

This work is part of **IEOR 240 (Optimization Analysis) and IEOR 242A (Machine Learning Analytics)** at **UC Berkeley**, demonstrating the intersection of **machine learning** and **optimization-based decision-making** in real-world housing challenges.


## **Motivation**
California faces a **severe housing affordability crisis**, where housing costs significantly outpace income growth. This project aims to:
- **Predict housing costs** based on historical trends and socio-economic indicators.
- **Optimize housing selection** by balancing affordability, location, and quality of life.
- **Support decision-making** for policymakers and individuals navigating the housing market.


## **Objectives**
1. **Predictive Modeling**  
   - Develop machine learning models (e.g., **Random Forest, Gradient Boosting**) to forecast **home prices** and **rental costs**.  
   - Identify key drivers of affordability using feature importance analysis.

2. **Optimization Framework**  
   - Formulate an **optimization model** to recommend the **best affordable housing options** based on income, location, and amenities.  
   - Use **constraint-based optimization** to maximize **Quality of Life (QoL)** metrics while ensuring affordability.  
   - Compare **Mixed Integer Linear Programming (MILP)** with **heuristic algorithms** for scalable solutions.

3. **Scenario Analysis & Policy Impact**  
   - Test **different affordability scenarios** under varying income distributions.  
   - Analyze the impact of **zoning policies and rent control measures** on affordability.  


## **Key Features**
### **1️⃣ Data-Driven Insights**
- Uses **2014–2024 housing market data** from authoritative sources:
  - **Zillow** (Home values & rental prices)
  - **U.S. Census Bureau** (Income & population density)
  - **Bureau of Labor Statistics** (Unemployment & wage data)
  - **California Department of Finance** (Economic indicators)
- Comprehensive dataset includes **housing, crime, air quality, mortgage rates, and healthcare accessibility**.

### **2️⃣ Predictive Modeling**
- **Regression & Machine Learning Models:**  
  - **Random Forest, Gradient Boosting, Ridge & LASSO Regression**
  - Feature selection for affordability drivers.

### **3️⃣ Optimization Models**
- **Goal:** Maximize **Quality of Life (QoL)** while ensuring affordability.
- **Constraints:**
  - Rent-to-income ratio limits.
  - Safety, healthcare accessibility, and environmental quality.
- **Methods:**
  - **Mixed Integer Linear Programming (MILP)**
  - **Simulated Annealing (SA)**
  - **Scenario-based optimization for policy evaluation**.


## **Project Workflow**
1. **Data Processing** (📂 `/src/data_processing.py`)  
   - Cleans & merges housing, economic, and demographic data.  
   - Produces **two datasets**:  
     - 📄 **`ml_data.csv`** → For predictive modeling (house-level data).  
     - 📄 **`optimization_data.csv`** → For metro-wide yearly optimization.

2. **Machine Learning Pipeline** (📂 `/src/ml_model.py`)  
   - Trains **Random Forest, Gradient Boosting, and Regression models**.  
   - Evaluates affordability prediction accuracy.

3. **Optimization Model** (📂 `/src/optimization.py`)  
   - Constructs a **Quality of Life (QoL) function** based on affordability, safety, and convenience.  
   - Runs **constrained optimization** to **recommend optimal locations**.


## **Project Structure**
```plaintext
cal-housing-optimization/
├── data/                  # Raw and processed datasets
│   ├── raw/               # Original data files
│   ├── processed/         # Cleaned datasets (full_data.csv, ml_data.csv, optimization_data.csv)
├── src/                   # Source code
│   ├── data_processing.py # Full ETL and preprocessing pipeline
│   ├── ml_model.py        # ML training, prediction, model loading
│   ├── optimization.py    # ILP + SA optimization models
├── main.py                # End-to-end pipeline
├── notebooks/             # Jupyter notebooks for EDA & choosing models
├── figures/               # Visualizations, logs, and summary tables
├── README.md              # Project documentation (this file)
└── LICENSE                # License information
