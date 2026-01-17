import streamlit as st
import pandas as pd
import process_data
import plotly.express as px

# Set page config
st.set_page_config(page_title="Aadhaar Lifecycle Intelligence", layout="wide", page_icon="🏛️")

# --- CUSTOM CSS FOR POLICY GRADE LOOK ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:20px !important;
        font-weight: 500;
    }
    .metric-box {
        padding: 10px;
        background-color: white;
        border-radius: 5px;
        border-left: 5px solid #4a90e2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #d1e4eb;
        margin-bottom: 20px;
    }
    .admin-header {
        color: #2c3e50;
        font-weight: bold;
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def get_data():
    biometric = process_data.load_dataset(process_data.BIO_DIR, "Biometric")
    demographic = process_data.load_dataset(process_data.DEMO_DIR, "Demographic")
    enrolment = process_data.load_dataset(process_data.ENROL_DIR, "Enrolment")
    
    biometric = process_data.clean_data(biometric)
    demographic = process_data.clean_data(demographic)
    enrolment = process_data.clean_data(enrolment)
    
    biometric, demographic, enrolment = process_data.feature_engineering(biometric, demographic, enrolment)
    
    # Calculate District Metrics (AUPS)
    district_metrics = process_data.calculate_district_metrics(biometric, enrolment)
    
    return biometric, demographic, enrolment, district_metrics

with st.spinner('Loading Policy-Grade Analytics System...'):
    bio_df, demo_df, enrol_df, metrics_df = get_data()

# --- HEADER ---
st.title("🏛️ Aadhaar Lifecycle Signal System")
st.markdown("**Policy-Grade Decision Intelligence System | Built for UIDAI Hackathon 2026**")
st.markdown("---")
# scope indicator moved down


# --- SIDEBAR FILTERS ---
st.sidebar.header("Operational Filters")
selected_state = st.sidebar.selectbox(
    "Select State for Analysis", 
    ["All"] + sorted(list(bio_df['state'].unique())) if not bio_df.empty else []
)

# scope indicator
st.subheader(f"Analytics Scope: {selected_state} level")

# --- RED FLAG ALERT SYSTEM ---
if selected_state != "All" and not metrics_df.empty:
    state_metrics = metrics_df[metrics_df['state'] == selected_state]
    critical_districts = state_metrics[state_metrics['AUPS_Normalized'] > 80]
    
    if not critical_districts.empty:
        count = len(critical_districts)
        dist_names = ", ".join(critical_districts['district'].head(3).tolist())
        st.error(f"🚨 **CRITICAL ALERT**: {count} districts in {selected_state} show Critical Update Pressure (AUPS > 80). Immediate intervention recommended for: {dist_names}...")
        with st.expander("View Critical Action Plan"):
            st.markdown("""
            **Recommended Administrative Interventions:**
            1. **Deploy Mobile Enrolment Kits (MEK)** to these districts within 48 hours.
            2. **Extend EC Operating Hours** to 8 PM.
            3. **Activate School-Cluster Camps** for mandatory biometric updates.
            """)

# Apply Filter
if selected_state != "All":
    bio_filtered = bio_df[bio_df['state'] == selected_state]
    demo_filtered = demo_df[demo_df['state'] == selected_state]
    enrol_filtered = enrol_df[enrol_df['state'] == selected_state]
    metrics_filtered = metrics_df[metrics_df['state'] == selected_state]
else:
    bio_filtered = bio_df
    demo_filtered = demo_df
    enrol_filtered = enrol_df
    metrics_filtered = metrics_df

# --- SECTION 1: LIFECYCLE TRANSITIONS (BIVARIATE ANALYSIS) ---
st.header(f"1. Lifecycle Transitions Analysis: {selected_state}")
st.caption("Bivariate Analysis: Update Volume vs Time | Purpose: Infrastructure Planning")

col1, col2 = st.columns([2, 1])

with col1:
    if not bio_filtered.empty and 'date' in bio_filtered.columns:
        # Aggregation
        daily_updates = bio_filtered.groupby('date')[['bio_age_5_17', 'bio_age_17_']].sum().reset_index()
        daily_updates.rename(columns={'bio_age_5_17': 'Child/Teen (Mandatory)', 'bio_age_17_': 'Adult (Voluntary)'}, inplace=True)
        
        # Rolling Average (7-day)
        daily_updates['Rolling Avg (Child)'] = daily_updates['Child/Teen (Mandatory)'].rolling(window=7, min_periods=1).mean()
        
        fig_time = px.line(daily_updates, x='date', y=['Child/Teen (Mandatory)', 'Rolling Avg (Child)', 'Adult (Voluntary)'], 
                           title="Daily Biometric Update Trends (with 7-day Rolling Avg)",
                           color_discrete_map={'Child/Teen (Mandatory)': '#3498db', 'Rolling Avg (Child)': '#2980b9', 'Adult (Voluntary)': '#95a5a6'})
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Insufficient data for time-series analysis.")

with col2:
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown('<div class="admin-header">Administrative Interpretation</div>', unsafe_allow_html=True)
    st.write("Spikes in the blue line (Child/Teen) typically correlate with school admission cycles or scholarship KYC deadlines.")
    
    st.markdown('<div class="admin-header">Operational Implication</div>', unsafe_allow_html=True)
    st.write("If rolling average exceeds baseline by >20% for 3 days, deploy mobile update kits to block-level schools immediately.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 2: HIGH PRESSURE ZONES (AUPS) ---
st.markdown("---")
st.header(f"2. High Pressure Zones (AUPS Ranking) - {selected_state}")
st.caption("Multivariate Analysis: Volume · Growth · Density | Purpose: Resource Allocation")

col3, col4 = st.columns([1, 2])

with col3:
    st.markdown("""
    ### Metric: AUPS
    **Aadhaar Update Pressure Score**
    
    A composite index (0-100) identifying districts under stress.
    
    $$AUPS = \\frac{Updates}{Enrolment} \\times (1 + GrowthRate)$$
    
    **Why it matters?**
    Identifies regions where demand is growing faster than base population (Migration/Compliance shock).
    """)
    
    st.warning("Districts with AUPS > 80 require immediate intervention.")

with col4:
    if not metrics_filtered.empty:
        top_districts = metrics_filtered.sort_values('AUPS_Normalized', ascending=False).head(10)
        
        fig_bar = px.bar(top_districts, x='AUPS_Normalized', y='district', orientation='h', 
                         title="Top 10 Districts by Update Pressure Score (AUPS)",
                         color='AUPS_Normalized', color_continuous_scale='Reds',
                         hover_data=['total_biometric_updates', 'growth_rate'])
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No metric data available.")



# --- SECTION 2B: COMPARATIVE STATE ANALYSIS ---
if selected_state != "All":
    st.markdown("---")
    st.header("2b. Comparative State Benchmarking")
    st.caption(f"Comparing {selected_state} vs Peer States | Purpose: Performance Standardization")
    
    # Logic to pick peers: Allow user to select reference state
    states = sorted(list(metrics_df['state'].unique()))
    default_ref = "Uttar Pradesh" if "Uttar Pradesh" in states and selected_state != "Uttar Pradesh" else (states[0] if states else None)
    
    ref_state = st.selectbox("Select Reference State for Comparison", states, index=states.index(default_ref) if default_ref in states else 0)
    
    # Calculate state-level AUPS avg
    state_aups = metrics_df.groupby('state')['AUPS_Normalized'].mean().reset_index()
    
    # Filter for Chart
    comp_data = state_aups[state_aups['state'].isin([selected_state, ref_state])]
    
    col_comp1, col_comp2 = st.columns(2)
    with col_comp1:
        st.metric(f"{selected_state} Avg AUPS", f"{comp_data[comp_data['state']==selected_state]['AUPS_Normalized'].values[0]:.1f}")
    with col_comp2:
        st.metric(f"{ref_state} (Reference) Avg AUPS", f"{comp_data[comp_data['state']==ref_state]['AUPS_Normalized'].values[0]:.1f}")
        
    fig_comp = px.bar(comp_data, x='state', y='AUPS_Normalized', title="State Stress Comparison", color='state')
    st.plotly_chart(fig_comp, use_container_width=True)
    
    st.info(f"**Insight**: If {selected_state}'s AUPS is significantly higher than {ref_state}, check for systemic backlog issues or migration waves.")
st.markdown("---")
st.header(f"3. Migration & Mobility Signals ({selected_state})")
st.caption("Trivariate Analysis: Enrolment (Valid) vs Update (Change) vs Geography")

if not demo_filtered.empty:
    total_new = enrol_filtered['age_0_5'].sum() + enrol_filtered['age_5_17'].sum() + enrol_filtered['age_18_greater'].sum()
    total_upd = demo_filtered['total_demographic_updates'].sum()
    adult_upd_share = demo_filtered['adult_update_share'].mean() * 100 if 'adult_update_share' in demo_filtered else 0
    
    col5, col6, col7 = st.columns(3)
    
    col5.metric("Total Demographic Updates", f"{total_upd:,}", delta="Changes in PII")
    col6.metric("Update-to-Enrolment Ratio", f"{(total_upd/total_new if total_new else 0):.2f}x", 
                delta="Signal Strength", delta_color="off")
    col7.metric("Adult Share of Updates", f"{adult_upd_share:.1f}%", help="High % = Economic Migration")
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown('<div class="admin-header">Policy Decision Point</div>', unsafe_allow_html=True)
    if (total_upd/total_new if total_new else 0) > 1.5:
        st.write(f"**Action Required**: This region is a **Mature Migration Hub** (Ratio > 1.5). Shift resources from Enrolment Centers (EC) to Update Centers (UC).")
    else:
        st.write(f"**Status Quo**: This region is still in **Net Growth Phase**. Maintain current Enrolment Center density.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# --- SECTION 4: FORECASTING & VALIDATION (DECISION INTELLIGENCE) ---
st.markdown("---")
st.header("4. Predictive Intelligence (Planning Module)")
st.caption("Non-ML Operational Forecast (30 Days) & Signal Validation")

col_fore1, col_fore2 = st.columns([2, 1])

with col_fore1:
    st.subheader("30-Day Demand Forecast")
    if not bio_filtered.empty:
        forecast_df = process_data.generate_forecast(bio_filtered, state=selected_state)
        if not forecast_df.empty:
            fig_cast = px.line(forecast_df, x='date', y='forecast', title="Projected Operational Demand (Next 30 Days)")
            fig_cast.add_scatter(x=forecast_df['date'], y=forecast_df['upper_ci'], mode='lines', 
                                 line=dict(width=0), showlegend=False, name='Upper Bound')
            fig_cast.add_scatter(x=forecast_df['date'], y=forecast_df['lower_ci'], mode='lines', 
                                 line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False, name='Confidence Band')
            st.plotly_chart(fig_cast, use_container_width=True)
        else:
            st.warning("Insufficient history for forecasting.")
    
with col_fore2:
    st.subheader("Signal Validation (Backtest)")
    with st.spinner("Running historical validation..."):
        validation = process_data.run_backtest_validation(bio_df, enrol_df)  # Use full data for robust check
        
    if validation.get('is_valid'):
        st.success("✅ **Model Validated**")
        st.write("Historical analysis confirms that High-AUPS districts consistently faced higher future demand.")
        st.metric("Predictive Lift", f"{validation.get('lift', 1):.2f}x", help="Districts signaled as High Pressure had this much more volume than normal districts in the next period.")
    else:
        st.warning("⚠️ **Validation Inconclusive**")
        st.write("Historical signal correlation is weak in current dataset.")

# --- FOOTER & METHODOLOGY ---
st.markdown("---")
with st.expander("Methodology & Data Dictionary"):
    st.markdown("""
    **Data Sources**: 
    - UIDAI Open Data API (Enrolment, Biometric Update, Demographic Update).
    
    **Formulas**:
    - **AUPS (Pressure Score)**: Combines update density (updates per enrollee) with temporal momentum (growth rate) to find hotspots.
    - **Forecasting**: 30-day extrapolation based on 7-day rolling trend (linear projection with uncertainty cone).
    - **Validation**: Split-half backtesting comparing T1 signals to T2 volume outcomes.
    
    **Limitations**:
    - Data is aggregated; individual PII is never accessed.
    - Forecasts are strictly operational scenarios, not machine learning predictions.
    """)

st.caption("© 2026 Aadhaar Data Innovation Hackathon Team. Code available for review.")
