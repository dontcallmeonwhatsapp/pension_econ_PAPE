import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from lee_carter_model import LeeCarter as lc

def parse_data(file_path):
    data = []
    current_year = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if not parts:
                continue
            
            # Check for year line (e.g., "1991/93")
            if re.match(r'^\d{4}/\d{2}$', parts[0]):
                current_year = int(parts[0][:4])
                current_sex = None
                continue
            
            if current_year is not None:
                if len(parts) > 2:
                    if parts[0] in ['Male', 'Female']:
                        current_sex = parts[0]
                    
                    if current_sex and parts[1] in ['55 years', '65 years', '75 years']:
                        try:
                            rate = float(parts[2])
                            data.append({'Year': current_year, 'Sex': current_sex, 'Age': parts[1], 'Mortality Rate': rate})
                        except ValueError:
                            continue
    return pd.DataFrame(data)

def prepare_lee_carter_data(df, sex):
    # Filter by sex
    subset = df[df['Sex'] == sex].copy()
    
    # Pivot to get Year x Age matrix
    # Ages are columns: 55 years, 65 years, 75 years
    pivot = subset.pivot(index='Year', columns='Age', values='Mortality Rate')
    
    # Sort columns to ensure consistent order (55, 65, 75)
    pivot = pivot[['55 years', '65 years', '75 years']]
    
    # Convert to list of lists of LOG mortality rates
    # LeeCarter.py expects log mortality rates for the input to lee_carter() function?
    # Let's check LeeCarter.py again.
    # mort_rates_db computes log rates.
    # lee_carter(log_mt) takes log rates.
    
    log_rates = np.log(pivot.values).tolist()
    years = pivot.index.tolist()
    
    return log_rates, years

def main():
    file_path = '12621-0001_en.csv'
    df = parse_data(file_path)
    
    ages = ['55 years', '65 years', '75 years']
    sexes = ['Male', 'Female']
    
    forecast_years = 2050 - 2022 # Forecast from last data point (2022) to 2050
    
    for sex in sexes:
        print(f"Forecasting for {sex}...")
        log_rates, years = prepare_lee_carter_data(df, sex)
        
        start_year = years[0]
        last_year = years[-1]
        n_forecast = 2050 - last_year
        
        # Run Lee-Carter Forecast manually to get parameters for Confidence Intervals
        # forecasts = lc.forc_lee_carter(log_rates, last_year + 1, n_forecast)
        
        # 1. Fit the model
        # lee_carter returns [ax, bx, kt]
        model_params = lc.lee_carter(log_rates)
        ax = model_params[0]
        bx = model_params[1]
        kt = model_params[2]
        
        # 2. Calculate Drift and Standard Error
        # Random Walk with Drift: kt = kt-1 + c + error
        # c (drift) = (k_T - k_1) / (T - 1)
        c = (kt[-1] - kt[0]) / (len(kt) - 1)
        
        # Calculate residuals to estimate standard deviation (sigma)
        residuals = []
        for t in range(1, len(kt)):
            k_t = kt[t]
            k_t_minus_1 = kt[t-1]
            # residual = actual_change - expected_change
            residuals.append((k_t - k_t_minus_1) - c)
        
        sigma_k = np.std(residuals, ddof=1) # Standard deviation of residuals
        
        # 3. Forecast kt with Confidence Intervals
        kt_forecast = []
        kt_lower = []
        kt_upper = []
        forc_years = []
        
        last_kt = kt[-1]
        
        for h in range(1, n_forecast + 1):
            year = last_year + h
            forc_years.append(year)
            
            # Point forecast: k_{T+h} = k_T + h * c
            k_h = last_kt + h * c
            kt_forecast.append(k_h)
            
            # Standard Error at horizon h: sigma * sqrt(h)
            se_h = sigma_k * np.sqrt(h)
            
            # 95% Confidence Interval (1.96 * SE)
            kt_lower.append(k_h - 1.96 * se_h)
            kt_upper.append(k_h + 1.96 * se_h)
            
        # 4. Convert back to Mortality Rates
        # ln(m_x) = a_x + b_x * k_t
        # m_x = exp(a_x + b_x * k_t)
        
        # Initialize dictionaries to hold lists of rates for each year
        forc_rates_point = {y: [] for y in forc_years}
        forc_rates_lower = {y: [] for y in forc_years}
        forc_rates_upper = {y: [] for y in forc_years}
        
        for t_idx, year in enumerate(forc_years):
            k_point = kt_forecast[t_idx]
            k_low = kt_lower[t_idx]
            k_up = kt_upper[t_idx]
            
            for x_idx in range(len(ax)):
                # Point forecast
                ln_mx = ax[x_idx] + bx[x_idx] * k_point
                forc_rates_point[year].append(np.exp(ln_mx))
                
                # Interval forecasts
                # Note: b_x can be positive or negative, so we need to be careful with upper/lower k_t
                # We calculate both and sort them to find true lower/upper bounds for m_x
                val1 = np.exp(ax[x_idx] + bx[x_idx] * k_low)
                val2 = np.exp(ax[x_idx] + bx[x_idx] * k_up)
                
                forc_rates_lower[year].append(min(val1, val2))
                forc_rates_upper[year].append(max(val1, val2))

        # Combine historical and forecast data for plotting
        
        # Historical data
        hist_years = years
        hist_rates = np.exp(log_rates) # Convert back to normal rates
        
        # Plotting
        plt.figure(figsize=(12, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
        
        for i, age in enumerate(ages):
            color = colors[i % len(colors)]
            
            # Historical
            hist_age_rates = [row[i] for row in hist_rates]
            plt.plot(hist_years, hist_age_rates, marker='o', color=color, label=f'Historical {age}')
            
            # Forecast Point
            forc_age_rates = [forc_rates_point[y][i] for y in forc_years]
            plt.plot(forc_years, forc_age_rates, linestyle='--', marker='x', color=color, label=f'Forecast {age}')
            
            # Confidence Interval
            lower_bound = [forc_rates_lower[y][i] for y in forc_years]
            upper_bound = [forc_rates_upper[y][i] for y in forc_years]
            
            plt.fill_between(forc_years, lower_bound, upper_bound, color=color, alpha=0.2, label=f'95% CI {age}')
            
        plt.title(f'Mortality Rate Forecast for {sex}s (Lee-Carter) with 95% CI')
        plt.xlabel('Year')
        plt.ylabel('Mortality Rate (q(x))')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_file = f'Forecast Plots/forecast_mortality_{sex.lower()}.png'
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close()

if __name__ == "__main__":
    main()
