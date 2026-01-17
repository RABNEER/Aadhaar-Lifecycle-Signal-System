import pandas as pd
import process_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def analyze_lifecycle_signals():
    # Load data using the processing script
    # We call the functions directly
    bio_df = process_data.load_dataset(process_data.BIO_DIR, "Biometric")
    demo_df = process_data.load_dataset(process_data.DEMO_DIR, "Demographic")
    enrol_df = process_data.load_dataset(process_data.ENROL_DIR, "Enrolment")
    
    # Process
    bio_df = process_data.clean_data(bio_df)
    demo_df = process_data.clean_data(demo_df)
    enrol_df = process_data.clean_data(enrol_df)
    
    bio_df, demo_df, enrol_df = process_data.feature_engineering(bio_df, demo_df, enrol_df)

    print("\nXXX ANALYSIS RESULTS XXX")

    # 1. Regions with High Lifecycle Transition Pressure (Biometric Updates)
    if not bio_df.empty:
        # Group by State
        state_bio = bio_df.groupby('state')['total_biometric_updates'].sum().sort_values(ascending=False).head(5)
        print("\n[INSIGHT] Top 5 States by Biometric Update Volume (Child -> Adult Transitions):")
        print(state_bio)
        
        # High pressure districts (normalize by something? simpler: just raw count for now)
        district_bio = bio_df.groupby(['state', 'district'])['total_biometric_updates'].sum().sort_values(ascending=False).head(5)
        print("\n[INSIGHT] Top 5 Districts by Biometric Update Pressure:")
        print(district_bio)

    # 2. Age Cohorts Analysis
    if not bio_df.empty:
        total_5_17 = bio_df['bio_age_5_17'].sum()
        total_17_plus = bio_df['bio_age_17_'].sum()
        total = total_5_17 + total_17_plus
        print(f"\n[INSIGHT] Biometric Update Composition:")
        print(f"  - Ages 5-17 (Mandatory Updates): {total_5_17:,} ({(total_5_17/total)*100:.1f}%)")
        print(f"  - Ages >17 (Other Updates): {total_17_plus:,} ({(total_17_plus/total)*100:.1f}%)")
        
    # 3. Mobility Signals (Demographic Updates)
    if not demo_df.empty:
        total_demo_updates = demo_df['total_demographic_updates'].sum()
        adult_demo_share = (demo_df['demo_age_17_'].sum() / total_demo_updates) * 100
        print(f"\n[INSIGHT] Demographic Update Mobility Signal:")
        print(f"  - Total Events: {total_demo_updates:,}")
        print(f"  - Adult Share (>17): {adult_demo_share:.1f}% (High share implies economic migration/correction)")

        # Top Mobility Districts
        mobility_districts = demo_df.groupby(['state', 'district'])['demo_age_17_'].sum().sort_values(ascending=False).head(5)
        print("\n[INSIGHT] Top 5 Districts for Adult Demographic Updates (Mobility Hotspots):")
        print(mobility_districts)
        
    # 4. Temporal Spikes
    if not bio_df.empty and 'date' in bio_df.columns:
        daily_updates = bio_df.groupby('date')['total_biometric_updates'].sum()
        peak_date = daily_updates.idxmax()
        peak_val = daily_updates.max()
        print(f"\n[INSIGHT] Peak Biometric Update Activity Date: {peak_date.date()} ({peak_val:,} updates)")

if __name__ == "__main__":
    analyze_lifecycle_signals()
