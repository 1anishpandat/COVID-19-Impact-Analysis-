"""
Generate COVID-19 Economic Impact Dataset
Simulates 60 months (2018-2023) across 8 industries

Run this FIRST before running the Jupyter notebook:
    python generate_covid_data.py

This will create: data/covid_economic_impact.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Date range: Jan 2018 - Dec 2022 (60 months)
start_date = datetime(2018, 1, 1)
months = pd.date_range(start=start_date, periods=60, freq='MS')

# 8 Industries with different COVID impacts
industries = {
    'Airlines': {
        'base_revenue': 2100,  # Million USD
        'covid_impact': -0.44,  # -44% decline
        'recovery_rate': 0.015,  # 1.5% recovery per month
        'volatility': 0.08
    },
    'E-commerce': {
        'base_revenue': 800,
        'covid_impact': 0.93,  # +93% growth!
        'recovery_rate': 0.02,  # Continued growth
        'volatility': 0.06
    },
    'Hospitality': {
        'base_revenue': 1500,
        'covid_impact': -0.38,  # -38% decline
        'recovery_rate': 0.018,
        'volatility': 0.10
    },
    'Healthcare': {
        'base_revenue': 1800,
        'covid_impact': 0.12,  # +12% growth (pandemic boost)
        'recovery_rate': 0.005,
        'volatility': 0.04
    },
    'Retail': {
        'base_revenue': 1200,
        'covid_impact': -0.25,  # -25% decline
        'recovery_rate': 0.012,
        'volatility': 0.07
    },
    'Technology': {
        'base_revenue': 1400,
        'covid_impact': 0.34,  # +34% growth (remote work boom)
        'recovery_rate': 0.01,
        'volatility': 0.05
    },
    'Manufacturing': {
        'base_revenue': 1600,
        'covid_impact': -0.18,  # -18% decline
        'recovery_rate': 0.008,
        'volatility': 0.06
    },
    'Financial Services': {
        'base_revenue': 1900,
        'covid_impact': -0.08,  # -8% slight decline
        'recovery_rate': 0.006,
        'volatility': 0.05
    }
}

# COVID timeline
COVID_START = 12  # March 2020 (index 26 in our date range)
COVID_PEAK = 15   # June 2020
VACCINE_START = 24  # December 2020

data = []

for industry_name, params in industries.items():
    base_revenue = params['base_revenue']
    covid_impact = params['covid_impact']
    recovery_rate = params['recovery_rate']
    volatility = params['volatility']
    
    for i, month_date in enumerate(months):
        # Determine period and calculate revenue
        months_since_2020 = i - 24  # Jan 2020 is month 24
        
        if months_since_2020 < 0:
            # Pre-COVID: Normal growth with seasonality
            revenue = base_revenue * (1 + 0.005 * i)  # 0.5% monthly growth
            period = 'Pre-COVID'
            covid_cases = 0
            
        elif 0 <= months_since_2020 < 3:
            # COVID onset (Jan-Mar 2020): Sharp decline starts
            impact_factor = 1 + (covid_impact * (months_since_2020 / 3))
            revenue = base_revenue * impact_factor
            period = 'COVID-Onset'
            covid_cases = 10000 * (months_since_2020 + 1) ** 2
            
        elif 3 <= months_since_2020 < 9:
            # Peak COVID (Apr-Sep 2020): Maximum impact
            revenue = base_revenue * (1 + covid_impact)
            period = 'COVID-Peak'
            covid_cases = 50000 + random.randint(-10000, 10000)
            
        elif 9 <= months_since_2020 < 15:
            # Vaccine rollout (Oct 2020 - Mar 2021): Start recovery
            months_recovering = months_since_2020 - 9
            recovery = covid_impact * (1 - months_recovering * recovery_rate * 2)
            revenue = base_revenue * (1 + recovery)
            period = 'Recovery'
            covid_cases = 40000 - (months_recovering * 2000)
            
        else:
            # Post-vaccine (Apr 2021+): Continued recovery
            months_recovering = months_since_2020 - 15
            recovery = covid_impact * max(0, 1 - 15 * recovery_rate * 2 - months_recovering * recovery_rate)
            revenue = base_revenue * (1 + recovery)
            period = 'Post-Vaccine'
            covid_cases = max(5000, 30000 - months_recovering * 1500)
        
        # Add seasonality (Q4 is stronger)
        month_num = month_date.month
        if month_num in [10, 11, 12]:  # Q4
            revenue *= 1.15
        elif month_num in [7, 8]:  # Summer slump
            revenue *= 0.95
        
        # Add random variation
        revenue *= (1 + np.random.normal(0, volatility))
        revenue = max(revenue, base_revenue * 0.2)  # Floor at 20% of base
        
        # Employment (correlates with revenue but lagged)
        employment = 1000 + (revenue / base_revenue) * 500
        employment += np.random.normal(0, 50)
        
        # Consumer spending (slightly different pattern)
        consumer_spending = revenue * 0.6 * (1 + np.random.normal(0, 0.1))
        
        # Unemployment rate (inverse of employment)
        unemployment_rate = max(3.5, 15 - (employment / 1500) * 10)
        
        data.append({
            'date': month_date,
            'year': month_date.year,
            'month': month_date.month,
            'quarter': f'Q{(month_date.month-1)//3 + 1}',
            'industry': industry_name,
            'revenue_million_usd': round(revenue, 2),
            'employment_thousands': round(employment, 1),
            'consumer_spending_million': round(consumer_spending, 2),
            'unemployment_rate_pct': round(unemployment_rate, 2),
            'covid_cases': int(covid_cases) if months_since_2020 >= 0 else 0,
            'period': period
        })

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values (realistic!)
missing_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
df.loc[missing_indices, 'consumer_spending_million'] = np.nan

missing_indices2 = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
df.loc[missing_indices2, 'employment_thousands'] = np.nan

# Save
df.to_csv('data/covid_economic_impact.csv', index=False)

print("=" * 70)
print("COVID-19 ECONOMIC IMPACT DATASET GENERATED")
print("=" * 70)
print(f"\n📊 Dataset Overview:")
print(f"   Records: {len(df):,}")
print(f"   Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"   Industries: {df['industry'].nunique()}")
print(f"   Time Period: {(df['date'].max() - df['date'].min()).days // 30} months")
print(f"\n📈 Revenue by Industry (Pre-COVID vs COVID-Peak):")

for industry in industries.keys():
    pre_covid = df[(df['industry']==industry) & (df['period']=='Pre-COVID')]['revenue_million_usd'].mean()
    covid_peak = df[(df['industry']==industry) & (df['period']=='COVID-Peak')]['revenue_million_usd'].mean()
    if pd.notna(pre_covid) and pd.notna(covid_peak):
        change_pct = ((covid_peak - pre_covid) / pre_covid) * 100
        print(f"   {industry:20} {change_pct:+7.1f}%")

