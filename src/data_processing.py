import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)
import pandas as pd
import numpy as np
import traceback

def load_datasets():
    """
    Load and merge all necessary datasets for both machine learning and optimization.
    Returns:
        pd.DataFrame: Fully merged dataset.
    """
    # Load datasets
    zhvi_data = pd.read_csv('../data/raw/ZHVI_2014-2024.csv')
    zori_data = pd.read_csv('../data/raw/ZORI_2015-2024.csv')
    agi_data = pd.read_csv('../data/raw/AGI_2014-2021.csv')
    crimes_data = pd.read_csv('../data/raw/Crimes_2014-2023.csv')
    mortgage_rates_data = pd.read_csv('../data/raw/Mortgage_Rates_2014-2024.csv')
    unemployment_rate_data = pd.read_csv('../data/raw/Unemployment_Rate_2014-2024.csv')
    census_data = pd.read_csv("../data/raw/census_data.csv")
    healthcare_data = pd.read_csv("../data/raw/healthcare_data.csv")
    air_quality = pd.read_csv('../data/raw/air_quality.csv')
    property_tax = pd.read_csv('../data/raw/property_tax.csv')

    # Merge zhvi and zori data
    merged_data = pd.merge(
        zhvi_data, 
        zori_data[['RegionID', 'RegionName', 'RegionType', 'StateName', 'City', 'Metro', 'CountyName', 'Date', 'ObservedRent']], 
        on=['RegionID', 'RegionName', 'RegionType', 'StateName', 'City', 'Metro', 'CountyName', 'Date'], 
        how='left'
    )

    # Convert 'Date' to datetime format and extract 'Year'
    merged_data['Date'] = pd.to_datetime(merged_data['Date'], errors='coerce')
    merged_data['Year'] = merged_data['Date'].dt.year

    # Standardize "Metro" column
    merged_data['Metro'] = merged_data['Metro'].str.replace(', CA', '')

    # Standardize column names and merge AGI data
    agi_data.rename(columns={'zipcode': 'RegionName'}, inplace=True)
    merged_data['RegionName'] = merged_data['RegionName'].astype(str)
    agi_data['RegionName'] = agi_data['RegionName'].astype(str)
    merged_data = pd.merge(merged_data, agi_data, on=['RegionName', 'Year'], how='left')

    # Merge crime data
    merged_data['CountyName'] = merged_data['CountyName'].str.strip()
    crimes_data['City'] = crimes_data['City'].str.strip()
    merged_data = pd.merge(merged_data, crimes_data, left_on=['CountyName', 'Year'], right_on=['City', 'Year'], how='left')
    merged_data.drop(columns=['City_y'], inplace=True)

    # Merge mortgage rates
    mortgage_rates_data['Date'] = pd.to_datetime(mortgage_rates_data['Date'], errors='coerce')
    merged_data = pd.merge(merged_data, mortgage_rates_data, on='Date', how='left')

    # Merge unemployment rate data
    merged_data['Month'] = merged_data['Date'].dt.month
    unemployment_rate_data.rename(columns={'Area': 'CountyName', 'Period': 'Month'}, inplace=True)
    unemployment_rate_data['Month'] = pd.to_datetime(unemployment_rate_data['Month'], format='%b').dt.month
    merged_data = pd.merge(merged_data, unemployment_rate_data, on=['CountyName', 'Year', 'Month'], how='left')

    # Merge census data
    census_data['ZIP'] = census_data['ZIP'].astype(str)
    merged_data = pd.merge(merged_data, census_data, left_on='RegionName', right_on='ZIP', how='left').drop(columns=['ZIP'])

    # Merge healthcare data
    healthcare_data['HealthCareFacilityAmmount'] = healthcare_data.groupby('ZIP')['ZIP'].transform('count')
    zip_facility_count = healthcare_data[['ZIP', 'HealthCareFacilityAmmount']].drop_duplicates().reset_index(drop=True)
    zip_facility_count['ZIP'] = zip_facility_count['ZIP'].astype(str)
    merged_data = pd.merge(merged_data, zip_facility_count, left_on='RegionName', right_on='ZIP', how='left').drop(columns=['ZIP'])
    merged_data['HealthCareFacilityAmmount'] = merged_data['HealthCareFacilityAmmount'].fillna(0).astype(int)

    # Merge air quality data
    air_quality = air_quality[air_quality['Year'] >= 2021].reset_index(drop=True)
    air_quality['Good_Percentage'] = air_quality['Good'] / air_quality['# Days with AQI']
    air_quality['Moderate_Percentage'] = air_quality['Moderate'] / air_quality['# Days with AQI']
    air_quality['Unhealthy_Sensitive_Percentage'] = air_quality['Unhealthy for Sensitive Groups'] / air_quality['# Days with AQI']
    air_quality['Unhealthy_Percentage'] = air_quality['Unhealthy'] / air_quality['# Days with AQI']
    air_quality['Very_Unhealthy_Percentage'] = air_quality['Very Unhealthy'] / air_quality['# Days with AQI']
    air_quality['Hazardous_Percentage'] = air_quality['Hazardous'] / air_quality['# Days with AQI']
    air_quality.drop(columns=['# Days with AQI', 'Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'], inplace=True)
    merged_data = pd.merge(merged_data, air_quality, how='left', left_on=['Metro', 'Year'], right_on=['CBSA', 'Year']).drop(columns=['CBSA'])
    
    # Merge property tax data
    property_tax = property_tax[property_tax['Fiscal Year'] >= 2021].reset_index(drop=True)
    merged_data = pd.merge(merged_data, property_tax, how='left', left_on=['City_x', 'Year'], right_on=['Entity Name', 'Fiscal Year']).drop(columns=['Entity Name', 'Fiscal Year'])

    return merged_data

def clean_data(df):
    """
    Clean dataset by handling missing values and standardizing columns.
    Args:
        df (pd.DataFrame): Merged dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    # Drop rows with missing Metro and population
    df = df.dropna(subset=['Metro', 'Total Population']).reset_index(drop=True)

    # Use interpolation for AGI
    df = df.sort_values(by=["RegionName", "Year"])
    df["agi_by_zipcode"] = df.groupby("RegionName")["agi_by_zipcode"].transform(lambda x: x.interpolate(method="linear"))
    county_year_avg = df.groupby(["CountyName", "Year"])["agi_by_zipcode"].transform("mean")
    df["agi_by_zipcode"] = df["agi_by_zipcode"].fillna(county_year_avg)

    # Fill crime data using median values
    for col in ['Arson Count', 'Violent Crimes Count', 'Property Crimes Count']:
        df[col] = df.groupby('CountyName')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df.groupby('Date')[col].transform(lambda x: x.fillna(x.median()))

    return df

def standardize_metro_names(df):
    """
    Standardize the naming of Metro areas to ensure consistency.
    Args:
        df (pd.DataFrame): Dataset containing the "Metro" column.
    Returns:
        pd.DataFrame: Dataset with standardized Metro names.
    """
    # Fix specific Metro name mappings
    df.loc[
        (df["Metro"] == "Sacramento-Roseville-Folsom") & (df["City_x"] == "Yolo"),
        "Metro"
    ] = "Yolo"

    df.loc[
        (df["Metro"] == "Sacramento-Roseville-Folsom") & (df["City_x"] != "Yolo"),
        "Metro"
    ] = "Sacramento-Roseville-Arden-Arcade"

    df.loc[
        (df["City_x"] == "San Francisco"),
        "Metro"
    ] = "San Francisco"

    df.loc[
        (df["Metro"] == "San Francisco-Oakland-Berkeley"),
        "Metro"
    ] = "Oakland-Fremont"

    df.loc[
        (df["City_x"].isin(["Santa Ana", "Anaheim", "Irvine"])),
        "Metro"
    ] = "Santa Ana-Anaheim-Irvine"

    df.loc[
        (df["Metro"] == "Los Angeles-Long Beach-Anaheim") & 
        (~df["City_x"].isin(["Santa Ana", "Anaheim", "Irvine"])),
        "Metro"
    ] = "Los Angeles-Long Beach-Glendale"

    df.loc[
        (df["CountyName"] == "San Benito County"),
        "Metro"
    ] = "San Benito County"

    return df

def prepare_ml_dataset(df):
    """
    Prepare dataset for the machine learning model.
    Keeps house-level data without aggregation.
    Args:
        df (pd.DataFrame): Preprocessed merged dataset.
    Returns:
        pd.DataFrame: Clean dataset for ML.
    """
    columns_needed = [
        "RegionID", "RegionName", "RegionType", "StateName", "City_x", "Metro", "CountyName",
        "Date", "HomeValue", "ObservedRent", "Year", "agi_by_zipcode",
        "Arson Count", "Property Crimes Count", "Violent Crimes Count",
        "30 yr FRM", "15 yr FRM", "Month", "Unemployment Rate",
        "Total Population", "Land Area in Square Miles", "Population Per Square Mile (Land Area)",
        "HealthCareFacilityAmmount"
    ]
    df_ml = df[columns_needed].copy()
    return df_ml

def prepare_optimization_dataset(df, hourly_wage):
    """
    Prepare dataset for the optimization model.
    Aggregates data to metro-level yearly from 2021 onwards.
    Args:
        df (pd.DataFrame): Preprocessed merged dataset.
        hourly_wage (pd.DataFrame): Hourly wage data.
    Returns:
        pd.DataFrame: Aggregated dataset for optimization.
    """
    df_opt = df[df['Year'] >= 2021].reset_index(drop=True)
    df_opt = df_opt.drop(columns=['RegionType', 'StateName', 'HomeValue', 'ObservedRent'])
    print(df_opt.head())
    # Standardize Metro names
    df_opt = standardize_metro_names(df_opt)

    # Fill air quality metrics with yearly averages
    air_quality_columns = ["Good_Percentage", "Moderate_Percentage", "Unhealthy_Sensitive_Percentage", "Unhealthy_Percentage", "Very_Unhealthy_Percentage", "Hazardous_Percentage"]
    for col in air_quality_columns:
        df_opt[col] = df_opt.groupby("Year")[col].transform(lambda x: x.fillna(x.mean()))

    # Fill tax values with yearly median
    tax_columns = ["Secured_Net Taxable Value", "Unsecured_Net Taxable Value", "Total Taxes Levied Countywide"]
    for col in tax_columns:
        df_opt[col] = df_opt.groupby("Year")[col].transform(lambda x: x.fillna(x.median()))
        
    # Group by Metro and Year
    agg_funcs = {
        "agi_by_zipcode": "mean",
        "Unemployment Rate": "mean",
        "Population Per Square Mile (Land Area)": "mean",
        "Arson Count": "sum",
        "Property Crimes Count": "sum",
        "Violent Crimes Count": "sum",
        "30 yr FRM": "mean",
        "15 yr FRM": "mean",
        'Good_Percentage': "mean", 
        'Moderate_Percentage': "mean",
        'Unhealthy_Sensitive_Percentage': "mean", 
        'Unhealthy_Percentage': "mean",
        'Very_Unhealthy_Percentage': "mean", 
        'Hazardous_Percentage': "mean",
        "Secured_Net Taxable Value": "sum",
        "Unsecured_Net Taxable Value": "sum",
        "Total Taxes Levied Countywide": "sum",
        "Total Population": "sum",
        "Land Area in Square Miles": "sum",
        "HealthCareFacilityAmmount": "sum",
    }
    grouped_df = df_opt.groupby(["Metro", "Year"]).agg(agg_funcs).reset_index()

    # Merge with Hourly Wage Data
    hourly_wage_long = pd.melt(hourly_wage, id_vars=["Metro"], var_name="Year", value_name="Hourly_Wage")
    hourly_wage_long["Year"] = hourly_wage_long["Year"].astype(int)
    final_merged_df = pd.merge(grouped_df, hourly_wage_long, on=["Metro", "Year"], how="right")

    return final_merged_df

def preprocess_pipeline(output_ml_path="../data/processed/ml_data.csv", output_opt_path="../data/processed/optimization_data.csv"):
    """
    Full pipeline: Load, clean, and return preprocessed datasets for ML and optimization.
    Args:
        output_ml_path (str): Path to save the ML dataset.
        output_opt_path (str): Path to save the optimization dataset.
    """
    print("Loading datasets...")
    df = load_datasets()

    print("Cleaning dataset...")
    df = clean_data(df)

    print("Generating Machine Learning dataset...")
    df_ml = prepare_ml_dataset(df)
    df_ml.to_csv(output_ml_path, index=False)
    print(f"ML dataset saved to {output_ml_path}")

    print("Generating Optimization dataset...")
    hourly_wage = pd.read_excel("../data/raw/wage_housing.xlsx", sheet_name='Data')

    df_opt = prepare_optimization_dataset(df, hourly_wage)
    df_opt.to_csv(output_opt_path, index=False)
    print(f"Optimization dataset saved to {output_opt_path}")

    return df_ml, df_opt

if __name__ == "__main__":
    print("Running data preprocessing pipeline...")

    try:
        df_ml, df_opt = preprocess_pipeline()

        print("\nML Dataset Preview:")
        print(df_ml.head())

        print("\nOptimization Dataset Preview:")
        print(df_opt.head())

        print("\nData preprocessing completed successfully!")

    except Exception as e:
        print("\nError occurred during data processing:")
        traceback.print_exc()