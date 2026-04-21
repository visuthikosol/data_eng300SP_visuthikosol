# Homework 1: Aircraft Inventory Analysis
**DATA ENG 300 | Nick Visuthikosol**

## How to Run the Code

1. Open `visuthikosol_HW1.ipynb` 
2. Run the first cell to install libraries
3. Run the second cell to download the dataset.
4. Run all remaining cells top to bottom

## Expected Outputs

### Task 1 — Missing Data
- Missing value summary table for 6 columns (CARRIER, CARRIER_NAME, AIRLINE_ID, MANUFACTURE_YEAR, NUMBER_OF_SEATS, CAPACITY_IN_POUNDS)
- After imputation: all 6 columns show 0 nulls except NUMBER_OF_SEATS (8,399) and CAPACITY_IN_POUNDS (5,949) 

### Task 2 — Standardization
- MANUFACTURER: consolidated from 183 variants down to 25 
- MODEL: 1,269 distinct models after cleaning
- AIRCRAFT_STATUS: merged to 4 lowercase codes (o, b, a, l)
- OPERATING_STATUS: mapped to Operating (126,648), Not Operating (5,664), Unknown (1)

### Task 3 — Drop Missing
- Rows removed: 43,427
- Rows remaining: 88,886

### Task 4 — Box-Cox Transformation
- Skewness before: NUMBER_OF_SEATS = -0.261, CAPACITY_IN_POUNDS = 2.002
- Skewness after: NUMBER_OF_SEATS_BOXCOX = -0.67, CAPACITY_IN_POUNDS_BOXCOX = 0.041
- 2 histograms before transformation
- 2 histograms after transformation (orange)

### Task 5 — Feature Engineering
- SIZE column created from NUMBER_OF_SEATS quartiles (Q1=50, Q2=117, Q3=154)
- Bar chart: Operating Status by Aircraft Size
- Bar chart: Aircraft Status by Size Group

### Task 6 — Modeling 
RMSE table for 4 models

## AI Usage Note

**Tool used:** Claude (claude.ai)

**Key prompts:**
- "How do I fill missing CARRIER values by looking up from CARRIER_NAME in the same dataframe?"
- "How do I impute AIRLINE_ID from CARRIER using a lookup?"
- "How do I apply scipy.stats.boxcox and save as a new column?"
- "Help me set up LinearRegression and RandomForestRegressor with train/test split and RMSE"
- Genreal Debugging and helping find errors in code

**What I changed and how I verified:**
- Adapted the carrier lookup function after discovering from my own value_counts() 
  exploration that North American Airlines was the only CARRIER null case
- Rewrote the AIRCRAFT_STATUS standardization entirely after seeing the actual 
  values in my data were letter codes (a, b, o, l), not the numeric 1/0 codes 
  Claude initially assumed
- Verified Box-Cox output by checking skewness values and visually inspecting histograms
- All model RMSE numbers were run and confirmed in Colab myself
