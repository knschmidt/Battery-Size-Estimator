import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Load data once at startup
@st.cache_data
def load_data():
    return pd.read_csv('results_df.csv')

results_df = load_data()

def calculate_savings_for_battery(battery_kwh, consumption_kwh, pv_kwh, leasing_fee_per_kwh, trawa_fee_per_kwh, df):
    """Calculate savings for a specific battery size"""
    # Convert to proportions for interpolation
    pv_proportion = pv_kwh / consumption_kwh
    battery_proportion = battery_kwh / consumption_kwh
    
    def extract_values(col_name):
        if 'con' in col_name and 'bat' in col_name and 'pv' in col_name:
            parts = col_name.split('_')
            con = float(parts[0].replace('con', ''))
            bat = float(parts[1].replace('bat', ''))
            pv = float(parts[2].replace('pv', ''))
            return pd.Series({'consumption_kwh': con, 'battery_proportion': bat, 'pv_proportion': pv})
        return pd.Series({'consumption_kwh': np.nan, 'battery_proportion': np.nan, 'pv_proportion': np.nan})

    point_data = pd.DataFrame([extract_values(col) for col in df.columns if 'con' in col])
    point_data = point_data.dropna()

    def get_interpolated_value(consumption, pv, battery, metric_series):
        points = point_data[['consumption_kwh', 'pv_proportion', 'battery_proportion']].values
        values = metric_series.values
        return griddata(points, values, np.array([[consumption, pv, battery]]), method='linear')[0]
    
    # Get costs data
    base_data = df.loc[df['index'] == 'annual_total_grid_load_cost'].iloc[0, 1:]
    fee_data = df.loc[df['index'] == 'annual_total_grid_fee_cost'].iloc[0, 1:]
    
    # Reference case (with PV but no battery)
    reference_grid_load_cost = get_interpolated_value(consumption_kwh, pv_proportion, 0, base_data)
    reference_grid_fee_cost = get_interpolated_value(consumption_kwh, pv_proportion, 0, fee_data)
    
    # Customer configuration (with battery)
    customer_grid_load_cost = get_interpolated_value(consumption_kwh, pv_proportion, battery_proportion, base_data)
    customer_grid_fee_cost = get_interpolated_value(consumption_kwh, pv_proportion, battery_proportion, fee_data)
    
    # Calculate savings
    grid_load_savings = reference_grid_load_cost - customer_grid_load_cost
    grid_fee_savings = reference_grid_fee_cost - customer_grid_fee_cost
    total_savings = grid_load_savings + grid_fee_savings
    
    # Calculate battery costs and net savings
    annual_battery_costs = battery_kwh * (leasing_fee_per_kwh + trawa_fee_per_kwh)
    net_savings = total_savings - annual_battery_costs
    
    # Calculate percentage savings including fees (compared to no-battery case)
    percentage_savings = (net_savings / reference_grid_load_cost) * 100
    
    return percentage_savings, net_savings, reference_grid_load_cost

def optimize_battery_size(consumption_kwh, pv_kwh, leasing_fee_per_kwh, trawa_fee_per_kwh):
    """Find optimal battery size that minimizes percentage savings"""
    # Define bounds for battery size
    max_battery = 0.0007 * consumption_kwh
    
    def objective_function(battery_kwh):
        percentage_savings, _, _ = calculate_savings_for_battery(
            battery_kwh, consumption_kwh, pv_kwh, leasing_fee_per_kwh, trawa_fee_per_kwh, results_df
        )
        return percentage_savings
    
    # Optimize battery size
    result = minimize_scalar(
        objective_function,
        bounds=(0, max_battery),
        method='bounded'
    )
    
    # Round to nearest whole kWh
    optimal_battery = round(result.x)
    
    # Calculate metrics with optimal battery
    percentage_savings, net_savings, reference_grid_load_cost = calculate_savings_for_battery(
        optimal_battery,
        consumption_kwh,
        pv_kwh,
        leasing_fee_per_kwh,
        trawa_fee_per_kwh,
        results_df
    )
    
    return {
        'optimal_battery_size_kwh': optimal_battery,
        'pv_size_kwh': pv_kwh,
        'consumption_kwh': consumption_kwh,
        'reference_case_grid_load_cost': reference_grid_load_cost,
        'estimated_annual_savings': net_savings,
        'percentage_total_grid_load_savings_incl_fees': percentage_savings
    }

# Streamlit interface
st.title('Battery Size Optimizer')

with st.form("optimizer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        consumption_kwh = st.number_input('Annual Consumption (kWh)', min_value=0)
        pv_kwh = st.number_input('PV System Size (kWh)', min_value=0)
    
    with col2:
        leasing_fee = st.number_input('Leasing Fee (€/kWh)', min_value=0.0)
        trawa_fee = st.number_input('Trawa Fee (€/kWh)', min_value=0.0)
    
    submitted = st.form_submit_button("Calculate Optimal Battery Size")

if submitted and consumption_kwh > 0:
    with st.spinner('Calculating optimal battery size...'):
        # Calculate optimal size
        results = optimize_battery_size(consumption_kwh, pv_kwh, leasing_fee, trawa_fee)
        
        # Display main results
        st.subheader('Optimization Results')
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Optimal Battery Size", f"{results['optimal_battery_size_kwh']:,.0f} kWh")
        with col2:
            st.metric("Annual Savings", f"€{results['estimated_annual_savings']:,.2f}")
        with col3:
            st.metric("Percentage Savings", f"{results['percentage_total_grid_load_savings_incl_fees']:.2f}%")
        
        # Comparison Analysis
        st.subheader('Comparison Analysis')
        optimal_size = results['optimal_battery_size_kwh']
        
        # Calculate metrics for nearby battery sizes
        battery_sizes = [
            max(0, optimal_size - 200),
            max(0, optimal_size - 100),
            optimal_size,
            optimal_size + 100,
            optimal_size + 200
        ]
        
        comparison_data = []
        for size in battery_sizes:
            percentage, savings, _ = calculate_savings_for_battery(
                size, consumption_kwh, pv_kwh, leasing_fee, trawa_fee, results_df
            )
            comparison_data.append({
                'Battery Size (kWh)': f"{size:,.0f}",
                'Annual Savings (€)': f"{savings:,.2f}",
                'Percentage Savings (%)': f"{percentage:.2f}"
            })
        
        # Display comparison table
        st.table(pd.DataFrame(comparison_data))
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        savings_values = [float(d['Annual Savings (€)'].replace(',', '')) for d in comparison_data]
        percentage_values = [float(d['Percentage Savings (%)']) for d in comparison_data]
        
        # Plot Annual Savings
        ax1.bar(range(len(battery_sizes)), savings_values, color='skyblue')
        ax1.set_xticks(range(len(battery_sizes)))
        ax1.set_xticklabels([d['Battery Size (kWh)'] for d in comparison_data], rotation=45)
        ax1.set_title('Annual Savings Comparison')
        ax1.set_xlabel('Battery Size (kWh)')
        ax1.set_ylabel('Annual Savings (€)')
        
        # Plot Percentage Savings
        ax2.bar(range(len(battery_sizes)), percentage_values, color='lightgreen')
        ax2.set_xticks(range(len(battery_sizes)))
        ax2.set_xticklabels([d['Battery Size (kWh)'] for d in comparison_data], rotation=45)
        ax2.set_title('Percentage Savings Comparison')
        ax2.set_xlabel('Battery Size (kWh)')
        ax2.set_ylabel('Percentage Savings (%)')
        
        plt.tight_layout()
        st.pyplot(fig)