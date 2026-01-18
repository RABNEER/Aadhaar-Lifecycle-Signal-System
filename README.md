# Public Service Demand Forecasting System (AUPS-Based)

## Overview
This project is a data-driven decision support system designed to forecast public service demand at National Level. It helps governments and administrators prevent overcrowding, reduce wait times, and optimize resource allocation using simple yet powerful analytics (non-ML).

## Key Features
- **AUPS (Area Utilization Pressure Score):** Quantifies service pressure at district level  
- **Red Flag Alert System 🚩:** Automatically flags districts when AUPS crosses 80  
- **Comparative State Analysis:** Benchmarks Arunachal Pradesh against other states  
- **Backtesting Validation:** Compares predicted demand with historical outcomes  
- **Short-Term Forecasting:** Uses 7-day rolling average to forecast next 30 days  
- **Edge Case Handling:** Detects districts with high AUPS but declining population  
- **Policy-Ready Insights:** Clear metrics suitable for reports and governance reviews  

## Why This Matters
- Prevents service center overcrowding  
- Improves citizen experience  
- Enables proactive planning instead of reactive fixes  
- Works without complex machine learning models  

## Methodology
1. Historical demand and population data ingestion  
2. AUPS calculation per district  
3. Rolling average-based short-term forecasting  
4. Backtesting to validate prediction accuracy  
5. Alert generation for high-risk districts  

## How to Run the Project

### 1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git  
cd your-repo-name  

### 2. (Optional) Create Virtual Environment
python -m venv venv  
source venv/bin/activate   # Windows: venv\Scripts\activate  

### 3. Install Dependencies
'''bash
pip install -r requirements.txt
'''

### 4. Add Dataset
Place all required CSV files inside the `data/` directory.

### 5. Run the Dashboard
streamlit run app.py  

### 6. Generate Final PDF Report
python generate_report.py 

## Outputs
- District-level demand forecasts  
- Red Flag alerts  
- Comparative dashboards  
- Policy-ready PDF reports  

## Use Cases
- Government planning departments  
- Smart governance initiatives  
- Hackathons and innovation challenges  
- Public service optimization  

## Tech Stack
- Python  
- Pandas / NumPy  
- Data visualization tools  

## Project Status
✅ Complete and submission-ready  
📄 PDF report preparation in progress  

---

*Built for impact, clarity, and real-world governance use.*
