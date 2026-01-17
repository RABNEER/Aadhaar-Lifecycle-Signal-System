import pandas as pd
import glob
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for paths
BASE_DIR = r"c:\Users\admin\OneDrive\Documents\Aadhaar as a Lifecycle Signal System"
BIO_DIR = os.path.join(BASE_DIR, "api_data_aadhar_biometric")
DEMO_DIR = os.path.join(BASE_DIR, "api_data_aadhar_demographic")
ENROL_DIR = os.path.join(BASE_DIR, "api_data_aadhar_enrolment")

def load_dataset(directory_path, name):
    """
    Loads all CSV files from a directory into a single DataFrame.
    """
    logging.info(f"Loading {name} data from {directory_path}...")
    files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not files:
        logging.warning(f"No CSV files found in {directory_path}")
        return pd.DataFrame()
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Loaded {name}: {combined_df.shape[0]} rows")
    return combined_df

def clean_data(df):
    """
    Standardizes schema and types.
    """
    if df.empty:
        return df
    
    # Standardize column names (lowercase, strip spaces)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Convert date to datetime
    if 'date' in df.columns:
        # Handles DD-MM-YYYY format based on inspection
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
    return df

def feature_engineering(bio_df, demo_df, enrol_df):
    """
    Creates lifecycle signal features.
    """
    logging.info("Starting feature engineering...")
    
    # 1. Lifecycle Signals in Biometric Data
    # 'bio_age_5_17' heavily implies the mandatory 5yo and 15yo updates.
    # We can rename/derivitize this for clarity.
    if not bio_df.empty:
        # Create a simplified total metric
        bio_df['total_biometric_updates'] = bio_df['bio_age_5_17'] + bio_df['bio_age_17_']
        # Cohort Share: What % of updates are from children/teens (lifecycle checks)?
        bio_df['child_teen_update_share'] = bio_df['bio_age_5_17'] / bio_df['total_biometric_updates']
    
    # 2. Mobility Signals in Demographic Data
    # 'demo_age_17_' implies adults changing details (often address/mobile).
    if not demo_df.empty:
        demo_df['total_demographic_updates'] = demo_df['demo_age_5_17'] + demo_df['demo_age_17_']
        # High adult share might indicate labor mobility vs child correction
        demo_df['adult_update_share'] = demo_df['demo_age_17_'] / demo_df['total_demographic_updates']

    # 3. Aggregations (Optional: Enrolment Ratios)
    # Merging typically requires aggregation by Region + Month to be meaningful, 
    # as daily data might be sparse or mismatched.
    
    return bio_df, demo_df, enrol_df

def calculate_district_metrics(bio_df, enrol_df):
    """
    Aggregates data by district to calculate:
    1. Total Biometric Updates
    2. Total Enrolments
    3. Biometric Update Growth Rate (Last 50% vs First 50% of time)
    4. AUPS (Aadhaar Update Pressure Score)
    """
    logging.info("Calculating District Metrics (AUPS)...")
    
    # 1. Aggregates
    bio_agg = bio_df.groupby(['state', 'district'])['total_biometric_updates'].sum().reset_index()
    enrol_agg = enrol_df.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'age_18_greater']].sum().reset_index()
    enrol_agg['total_enrolment'] = enrol_agg['age_0_5'] + enrol_agg['age_5_17'] + enrol_agg['age_18_greater']
    
    # Merge
    metrics = pd.merge(bio_agg, enrol_agg[['state', 'district', 'total_enrolment']], on=['state', 'district'], how='left')
    metrics['total_enrolment'] = metrics['total_enrolment'].fillna(1) # Avoid div by zero
    
    # 2. Growth Rate (Simplified: Last half vs First half of the dataset duration)
    # We do this per district.
    growth_rates = []
    
    # Ensure date is set
    if 'date' in bio_df.columns:
        bio_sorted = bio_df.sort_values('date')
        mid_date = bio_sorted['date'].mean() # Simple split point
        
        for (state, district), group in bio_sorted.groupby(['state', 'district']):
            recent = group[group['date'] > mid_date]['total_biometric_updates'].sum()
            past = group[group['date'] <= mid_date]['total_biometric_updates'].sum()
            
            if past > 0:
                growth = (recent - past) / past
            else:
                growth = 0.0
            
            growth_rates.append({'state': state, 'district': district, 'growth_rate': growth})
            
        growth_df = pd.DataFrame(growth_rates)
        metrics = pd.merge(metrics, growth_df, on=['state', 'district'], how='left')
    else:
        metrics['growth_rate'] = 0.0

    # 3. Calculate AUPS
    # Formula: (Updates / Enrolment) * (1 + Growth Rate)
    # We normalize updates/enrolment to handle scale issues and just score it.
    metrics['update_density'] = metrics['total_biometric_updates'] / metrics['total_enrolment']
    
    # Normalizing growth rate for the score (map 0 to 1, doubling growth = x2 score)
    # We use (1 + growth_rate) as the multiplier.
    metrics['growth_multiplier'] = 1 + metrics['growth_rate'].clip(lower=-0.5, upper=2.0)
    
    metrics['AUPS'] = metrics['update_density'] * metrics['growth_multiplier'] * 1000 # Scale up for readability
    
    metrics = metrics.sort_values('AUPS', ascending=False)
    
    # Normalize AUPS to 0-100 for score presentation
    max_score = metrics['AUPS'].max()
    if max_score > 0:
        metrics['AUPS_Normalized'] = (metrics['AUPS'] / max_score) * 100
    else:
        metrics['AUPS_Normalized'] = 0
        
    return metrics

def generate_forecast(df, state="All", days=30):
    """
    Generates a 30-day operational planning forecast using 7-day rolling trends.
    Returns DataFrame with [date, forecast, lower_bound, upper_bound]
    """
    logging.info(f"Generating forecast for {state}...")
    
    if state != "All":
        df = df[df['state'] == state]
    
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()

    # Aggregate daily
    daily = df.groupby('date')['total_biometric_updates'].sum().reset_index()
    daily = daily.sort_values('date')
    
    # Simple Trend: Last 14 days avg growth
    last_14 = daily.tail(14)
    if len(last_14) < 14:
        return pd.DataFrame() # Not enough data
        
    avg_growth = last_14['total_biometric_updates'].pct_change().mean()
    last_val = last_14['total_biometric_updates'].iloc[-1]
    last_date = last_14['date'].iloc[-1]
    
    forecast_data = []
    current_val = last_val
    
    # Cap insane growth for stability (conservative planning)
    growth_factor = 1 + max(min(avg_growth, 0.05), -0.05) 
    
    for i in range(1, days + 1):
        next_date = last_date + pd.Timedelta(days=i)
        current_val = current_val * growth_factor
        
        # Confidence Band (Widens over time)
        uncertainty = 0.05 + (0.01 * i) # Starts at 5%, adds 1% per day
        
        forecast_data.append({
            'date': next_date,
            'forecast': int(current_val),
            'lower_ci': int(current_val * (1 - uncertainty)),
            'upper_ci': int(current_val * (1 + uncertainty))
        })
        
    return pd.DataFrame(forecast_data)

def run_backtest_validation(bio_df, enrol_df):
    """
    Validates if High AUPS signals actually predict higher update demand.
    Method: Split time into T1 (Signal) and T2 (Outcome).
    """
    logging.info("Running Backtest Validation...")
    
    if 'date' not in bio_df.columns:
        return {"accuracy": 0, "desc": "No date column"}
        
    bio_sorted = bio_df.sort_values('date')
    mid_idx = len(bio_sorted) // 2
    
    t1_df = bio_sorted.iloc[:mid_idx]
    t2_df = bio_sorted.iloc[mid_idx:]
    
    if t1_df.empty or t2_df.empty:
        return {"accuracy": 0, "desc": "Insufficient data split"}
        
    # Calculate AUPS for T1
    t1_metrics = calculate_district_metrics(t1_df, enrol_df)
    
    # Identify Top 20% High Pressure Districts in T1
    threshold = t1_metrics['AUPS'].quantile(0.8)
    stressed_districts = t1_metrics[t1_metrics['AUPS'] >= threshold][['state', 'district']]
    
    # Check T2 Volume for these districts vs Normal districts
    # Merge T2 agg with stressed list
    t2_agg = t2_df.groupby(['state', 'district'])['total_biometric_updates'].sum().reset_index()
    t2_agg = t2_agg.merge(stressed_districts, on=['state', 'district'], how='left', indicator=True)
    
    t2_stressed_vol = t2_agg[t2_agg['_merge'] == 'both']['total_biometric_updates'].mean()
    t2_normal_vol = t2_agg[t2_agg['_merge'] == 'left_only']['total_biometric_updates'].mean()
    
    # Directional Accuracy: Did Stressed districts have higher volume in T2?
    is_valid = t2_stressed_vol > t2_normal_vol
    
    # Simple score
    lift = (t2_stressed_vol / t2_normal_vol) if t2_normal_vol > 0 else 1.0
    
    return {
        "is_valid": is_valid,
        "stressed_avg_t2": t2_stressed_vol,
        "normal_avg_t2": t2_normal_vol,
        "lift": lift
    }

def main():
    # Load
    bio_df = load_dataset(BIO_DIR, "Biometric")
    demo_df = load_dataset(DEMO_DIR, "Demographic")
    enrol_df = load_dataset(ENROL_DIR, "Enrolment")
    
    # Clean
    bio_df = clean_data(bio_df)
    demo_df = clean_data(demo_df)
    enrol_df = clean_data(enrol_df)
    
    # Feature Engineering
    bio_df, demo_df, enrol_df = feature_engineering(bio_df, demo_df, enrol_df)
    
    # Preview
    print("\n--- Biometric Data Preview ---")
    print(bio_df.head())
    print(bio_df.info())
    
    print("\n--- Demographic Data Preview ---")
    print(demo_df.head())
    
    print("\n--- Enrolment Data Preview ---")
    print(enrol_df.head())

    # Return logical datasets for import by app.py
    return {
        'biometric': bio_df,
        'demographic': demo_df,
        'enrolment': enrol_df
    }

if __name__ == "__main__":
    main()
