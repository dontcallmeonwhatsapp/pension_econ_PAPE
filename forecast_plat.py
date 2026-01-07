"""
Plat Model Implementation for Mortality Forecasting
Reference: Plat, R. (2009). On Stochastic Mortality Modeling. 
           Insurance: Mathematics and Economics, 45(3), 393-404.

The Plat model combines features of Lee-Carter and CBD models:
ln(m_{x,t}) = α_x + κ_t^(1) + (x-x̄)κ_t^(2) + (x̄-x)^+ κ_t^(3) + γ_{t-x}

Where (x̄-x)^+ = max(x̄-x, 0), capturing additional curvature at younger ages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import re
import warnings
warnings.filterwarnings('ignore')

def parse_data(file_path):
    """Parse mortality data from CSV file."""
    data = []
    current_year = None
    current_sex = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if not parts:
                continue
            
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
                            data.append({
                                'Year': current_year, 
                                'Sex': current_sex, 
                                'Age': parts[1], 
                                'Mortality Rate': rate
                            })
                        except ValueError:
                            continue
    return pd.DataFrame(data)

def prepare_data(df, sex):
    """Prepare data matrix for model fitting."""
    subset = df[df['Sex'] == sex].copy()
    pivot = subset.pivot(index='Year', columns='Age', values='Mortality Rate')
    pivot = pivot[['55 years', '65 years', '75 years']]
    log_rates = np.log(pivot.values)
    years = pivot.index.tolist()
    ages = [55, 65, 75]
    return log_rates, years, ages

def fit_plat_model(log_rates, years, ages):
    """
    Fit the Plat model using iterative least squares.
    
    The Plat model: ln(m_{x,t}) = α_x + κ_t^(1) + (x-x̄)κ_t^(2) + (x̄-x)^+ κ_t^(3) + γ_{t-x}
    
    For simplicity with only 3 ages, we use a reduced form focusing on:
    - α_x: age effect
    - κ_t^(1): level (like Lee-Carter k_t)
    - κ_t^(2): slope (like CBD)
    - γ_{c}: cohort effect where c = t - x
    """
    n_years, n_ages = log_rates.shape
    x_bar = np.mean(ages)
    
    # Initialize parameters
    alpha_x = np.mean(log_rates, axis=0)  # Average log mortality by age
    
    # Center the data
    centered = log_rates - alpha_x
    
    # Extract period effects using regression approach
    # For each year t: centered[t,:] = κ1[t] + (x - x̄) * κ2[t] + (x̄ - x)^+ * κ3[t]
    
    kappa1 = np.zeros(n_years)
    kappa2 = np.zeros(n_years)
    kappa3 = np.zeros(n_years)
    
    # Design matrix for age effects
    X = np.zeros((n_ages, 3))
    for i, x in enumerate(ages):
        X[i, 0] = 1  # Intercept for κ1
        X[i, 1] = x - x_bar  # Slope for κ2
        X[i, 2] = max(x_bar - x, 0)  # Young age effect for κ3
    
    # Fit period effects for each year
    for t in range(n_years):
        y = centered[t, :]
        # Least squares: (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            kappa1[t] = beta[0]
            kappa2[t] = beta[1]
            kappa3[t] = beta[2]
        except:
            kappa1[t] = np.mean(y)
            kappa2[t] = 0
            kappa3[t] = 0
    
    # Extract cohort effects from residuals
    residuals = np.zeros_like(log_rates)
    for t in range(n_years):
        for i, x in enumerate(ages):
            fitted = alpha_x[i] + kappa1[t] + (x - x_bar) * kappa2[t] + max(x_bar - x, 0) * kappa3[t]
            residuals[t, i] = log_rates[t, i] - fitted
    
    # Cohort effects: group residuals by cohort (year - age)
    cohorts = {}
    for t_idx, year in enumerate(years):
        for a_idx, age in enumerate(ages):
            cohort = year - age
            if cohort not in cohorts:
                cohorts[cohort] = []
            cohorts[cohort].append(residuals[t_idx, a_idx])
    
    gamma = {c: np.mean(vals) for c, vals in cohorts.items()}
    
    # Normalize: sum of kappa1 = 0, sum of gamma = 0
    kappa1_mean = np.mean(kappa1)
    kappa1 = kappa1 - kappa1_mean
    alpha_x = alpha_x + kappa1_mean
    
    gamma_mean = np.mean(list(gamma.values()))
    gamma = {c: g - gamma_mean for c, g in gamma.items()}
    alpha_x = alpha_x + gamma_mean / n_ages
    
    return {
        'alpha_x': alpha_x,
        'kappa1': kappa1,
        'kappa2': kappa2,
        'kappa3': kappa3,
        'gamma': gamma,
        'ages': ages,
        'years': years,
        'x_bar': x_bar
    }

def forecast_plat(model, n_forecast):
    """
    Forecast mortality using the fitted Plat model.
    
    Time series models:
    - κ1: Random walk with drift
    - κ2: Random walk (no drift, as slope changes are mean-reverting)
    - κ3: Random walk with drift
    - γ: Extrapolate for new cohorts using ARIMA(1,1,0)
    """
    kappa1 = model['kappa1']
    kappa2 = model['kappa2']
    kappa3 = model['kappa3']
    gamma = model['gamma']
    alpha_x = model['alpha_x']
    ages = model['ages']
    years = model['years']
    x_bar = model['x_bar']
    
    last_year = years[-1]
    n_years = len(years)
    
    # Drift and volatility for κ1
    dk1 = np.diff(kappa1)
    drift_k1 = np.mean(dk1)
    sigma_k1 = np.std(dk1, ddof=1)
    
    # Drift and volatility for κ2
    dk2 = np.diff(kappa2)
    drift_k2 = np.mean(dk2)
    sigma_k2 = np.std(dk2, ddof=1)
    
    # Drift and volatility for κ3
    dk3 = np.diff(kappa3)
    drift_k3 = np.mean(dk3)
    sigma_k3 = np.std(dk3, ddof=1)
    
    # Cohort effect extrapolation
    sorted_cohorts = sorted(gamma.keys())
    gamma_values = [gamma[c] for c in sorted_cohorts]
    
    # Simple AR(1) for cohort extrapolation
    if len(gamma_values) > 2:
        dg = np.diff(gamma_values)
        drift_g = np.mean(dg)
        sigma_g = np.std(dg, ddof=1) if len(dg) > 1 else 0.01
    else:
        drift_g = 0
        sigma_g = 0.01
    
    # Forecast
    forecast_years = list(range(last_year + 1, last_year + n_forecast + 1))
    
    results = {
        'years': forecast_years,
        'rates_point': {age: [] for age in ages},
        'rates_lower': {age: [] for age in ages},
        'rates_upper': {age: [] for age in ages}
    }
    
    for h, year in enumerate(forecast_years, 1):
        # Forecast period effects
        k1_h = kappa1[-1] + h * drift_k1
        k2_h = kappa2[-1] + h * drift_k2
        k3_h = kappa3[-1] + h * drift_k3
        
        # Combined standard error (simplified)
        se_h = np.sqrt(h) * np.sqrt(sigma_k1**2 + sigma_k2**2 + sigma_k3**2)
        
        for i, age in enumerate(ages):
            # Cohort for this age in this year
            cohort = year - age
            
            # Extrapolate cohort effect if needed
            if cohort in gamma:
                g_c = gamma[cohort]
            else:
                # Extrapolate from last known cohort
                last_known = max(sorted_cohorts)
                steps = cohort - last_known
                g_c = gamma[last_known] + steps * drift_g
            
            # Point forecast
            ln_mx = (alpha_x[i] + k1_h + (age - x_bar) * k2_h + 
                     max(x_bar - age, 0) * k3_h + g_c)
            
            # Confidence intervals (95%)
            ln_mx_lower = ln_mx - 1.96 * se_h
            ln_mx_upper = ln_mx + 1.96 * se_h
            
            results['rates_point'][age].append(np.exp(ln_mx))
            results['rates_lower'][age].append(np.exp(ln_mx_lower))
            results['rates_upper'][age].append(np.exp(ln_mx_upper))
    
    return results

def calculate_goodness_of_fit(log_rates, model):
    """Calculate in-sample fit statistics."""
    ages = model['ages']
    years = model['years']
    alpha_x = model['alpha_x']
    kappa1 = model['kappa1']
    kappa2 = model['kappa2']
    kappa3 = model['kappa3']
    gamma = model['gamma']
    x_bar = model['x_bar']
    
    fitted = np.zeros_like(log_rates)
    for t_idx, year in enumerate(years):
        for a_idx, age in enumerate(ages):
            cohort = year - age
            g_c = gamma.get(cohort, 0)
            fitted[t_idx, a_idx] = (alpha_x[a_idx] + kappa1[t_idx] + 
                                    (age - x_bar) * kappa2[t_idx] +
                                    max(x_bar - age, 0) * kappa3[t_idx] + g_c)
    
    residuals = log_rates - fitted
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_rates - np.mean(log_rates))**2)
    r_squared = 1 - ss_res / ss_tot
    
    # Number of parameters (approximate)
    n_params = len(ages) + 3 * len(years) + len(gamma)
    n_obs = log_rates.size
    
    # AIC and BIC
    log_likelihood = -n_obs / 2 * (np.log(2 * np.pi) + np.log(ss_res / n_obs) + 1)
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_obs) * n_params - 2 * log_likelihood
    
    return {
        'rmse': rmse,
        'r_squared': r_squared * 100,
        'aic': aic,
        'bic': bic
    }

def main():
    print("="*60)
    print("PLAT MODEL - Mortality Forecasting")
    print("="*60)
    
    file_path = '12621-0001_en.csv'
    df = parse_data(file_path)
    
    ages_labels = ['55 years', '65 years', '75 years']
    ages = [55, 65, 75]
    sexes = ['Male', 'Female']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for sex in sexes:
        print(f"\n{'='*40}")
        print(f"Processing {sex}...")
        print('='*40)
        
        log_rates, years, ages = prepare_data(df, sex)
        
        # Fit Plat model
        model = fit_plat_model(log_rates, years, ages)
        
        # Calculate goodness of fit
        gof = calculate_goodness_of_fit(log_rates, model)
        print(f"\nGoodness of Fit Statistics:")
        print(f"  RMSE: {gof['rmse']:.4f}")
        print(f"  R²: {gof['r_squared']:.1f}%")
        print(f"  AIC: {gof['aic']:.0f}")
        print(f"  BIC: {gof['bic']:.0f}")
        
        # Forecast
        last_year = years[-1]
        n_forecast = 2050 - last_year
        forecast = forecast_plat(model, n_forecast)
        
        # Historical rates
        hist_rates = np.exp(log_rates)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        for i, age in enumerate(ages):
            color = colors[i]
            
            # Historical
            hist_age_rates = hist_rates[:, i]
            plt.plot(years, hist_age_rates, marker='o', color=color, 
                     label=f'Historical Age {age}', markersize=4)
            
            # Forecast point
            plt.plot(forecast['years'], forecast['rates_point'][age], 
                     linestyle='--', marker='x', color=color, 
                     label=f'Forecast Age {age}', markersize=4)
            
            # Confidence interval
            plt.fill_between(forecast['years'], 
                             forecast['rates_lower'][age],
                             forecast['rates_upper'][age],
                             color=color, alpha=0.2, label=f'95% CI Age {age}')
        
        plt.title(f'Plat Model Mortality Forecast for {sex}s (1991-2050)\nwith 95% Confidence Intervals', 
                  fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Mortality Rate (q(x))', fontsize=12)
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_file = f'Forecast Plots/forecast_plat_{sex.lower()}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to {output_file}")
        plt.close()
        
        # Print forecast values
        print(f"\nForecast mortality rates for {sex}s:")
        print(f"{'Year':<8} {'Age 55':<12} {'Age 65':<12} {'Age 75':<12}")
        print("-"*44)
        for key_year in [2023, 2030, 2040, 2050]:
            if key_year in forecast['years']:
                idx = forecast['years'].index(key_year)
                print(f"{key_year:<8} {forecast['rates_point'][55][idx]:<12.5f} "
                      f"{forecast['rates_point'][65][idx]:<12.5f} "
                      f"{forecast['rates_point'][75][idx]:<12.5f}")

if __name__ == "__main__":
    main()

