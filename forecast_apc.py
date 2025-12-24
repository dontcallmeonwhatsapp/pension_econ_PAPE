import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_german_data(file_path):
    data = []
    current_year = None
    current_sex = None
    
    # Mapping German to English
    sex_map = {'männlich': 'Male', 'weiblich': 'Female'}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            parts = line.split(';')
            
            if not parts or parts == ['']:
                continue
            
            # Check for year line (e.g., "1991/93")
            if re.match(r'^\d{4}/\d{2}$', parts[0]):
                current_year = int(parts[0][:4])
                current_sex = None
                continue
            
            # Check for sex line or data line
            # "männlich;0 Jahre;0,00698929"
            # ";1 Jahr;0,00058753"
            
            first_col = parts[0]
            
            if first_col in sex_map:
                current_sex = sex_map[first_col]
            
            # If we have a valid year and sex (either just set or carried over)
            if current_year is not None:
                # Determine where age and rate are
                # If line starts with sex: parts[0]=sex, parts[1]=age, parts[2]=rate
                # If line starts with empty: parts[0]='', parts[1]=age, parts[2]=rate
                
                age_str = ""
                rate_str = ""
                
                if len(parts) >= 3:
                    if parts[0] in sex_map:
                        current_sex = sex_map[parts[0]]
                        age_str = parts[1]
                        rate_str = parts[2]
                    elif parts[0] == '' and current_sex is not None:
                        age_str = parts[1]
                        rate_str = parts[2]
                
                if age_str and rate_str:
                    # Parse Age
                    # "0 Jahre", "1 Jahr", "90 Jahre"
                    age_match = re.match(r'(\d+)\s*Jahr', age_str)
                    if age_match:
                        age = int(age_match.group(1))
                        
                        # Parse Rate (handle German comma)
                        try:
                            rate = float(rate_str.replace(',', '.'))
                            data.append({
                                'Year': current_year,
                                'Sex': current_sex,
                                'Age': age,
                                'Mortality Rate': rate
                            })
                        except ValueError:
                            continue

    return pd.DataFrame(data)

def fit_apc_model(df, sex):
    # Filter by sex
    subset = df[df['Sex'] == sex].copy()
    
    # Pivot: Year x Age
    pivot = subset.pivot(index='Year', columns='Age', values='Mortality Rate')
    
    # Filter ages to a reasonable range (e.g., 0 to 90) to avoid noise at extreme old ages
    # The user wants 55, 65, 75, so we must include those.
    # Let's use 0-90.
    valid_ages = [a for a in pivot.columns if isinstance(a, (int, float)) and a <= 90]
    pivot = pivot[valid_ages]
    
    years = pivot.index.values

    ages = np.array(valid_ages)
    
    # Log Mortality Rates
    # Handle zeros if any (replace with small epsilon)
    mx = pivot.values
    # Fill NaNs with interpolation or just drop them?
    # Better to interpolate or fill.
    # For now, let's check if there are NaNs.
    if np.isnan(mx).any():
        print("Warning: Data contains NaNs. Filling with forward/backward fill.")
        pivot = pivot.ffill().bfill()
        mx = pivot.values
        
    mx[mx <= 0] = 1e-9
    ln_mx = np.log(mx)
    
    # Dimensions
    ny = len(years)
    nx = len(ages)
    
    # Initialize parameters
    # alpha_x: Average level of mortality at age x
    alpha_x = np.mean(ln_mx, axis=0)
    
    # kappa_t: Period effect (initially 0)
    kappa_t = np.zeros(ny)
    
    # gamma_c: Cohort effect
    # Cohort index c = t - x
    # We need to map (t, x) to a cohort index.
    # Min cohort = min(years) - max(ages)
    # Max cohort = max(years) - min(ages)
    min_cohort = years[0] - ages[-1]
    max_cohort = years[-1] - ages[0]
    cohorts = np.arange(min_cohort, max_cohort + 1)
    gamma_c = np.zeros(len(cohorts))
    
    # Helper to get cohort index from year_idx and age_idx
    def get_c_idx(y_idx, x_idx):
        y = years[y_idx]
        x = ages[x_idx]
        c = y - x
        return c - min_cohort

    # Iterative Fitting (Simple Newton-Raphson-like or Backfitting)
    # Model: ln_mx[t, x] = alpha_x[x] + kappa_t[t] + gamma_c[c]
    
    for _ in range(50): # Max iterations
        
        # 1. Update Kappa_t
        # kappa_t = mean_over_x(ln_mx - alpha_x - gamma_c)
        for t in range(ny):
            res_sum = 0
            count = 0
            for x in range(nx):
                c_idx = get_c_idx(t, x)
                res_sum += ln_mx[t, x] - alpha_x[x] - gamma_c[c_idx]
                count += 1
            kappa_t[t] = res_sum / count
        
        # Center Kappa (Constraint: sum(kappa) = 0)
        k_mean = np.mean(kappa_t)
        kappa_t -= k_mean
        alpha_x += k_mean # Absorb into alpha
        
        # 2. Update Gamma_c
        # gamma_c = mean_over_occurrences(ln_mx - alpha_x - kappa_t)
        gamma_sums = np.zeros(len(cohorts))
        gamma_counts = np.zeros(len(cohorts))
        
        for t in range(ny):
            for x in range(nx):
                c_idx = get_c_idx(t, x)
                resid = ln_mx[t, x] - alpha_x[x] - kappa_t[t]
                gamma_sums[c_idx] += resid
                gamma_counts[c_idx] += 1
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            new_gamma = gamma_sums / gamma_counts
            new_gamma[np.isnan(new_gamma)] = 0 # Unobserved cohorts
        
        gamma_c = new_gamma
        
        # Center Gamma (Constraint: sum(gamma) = 0, slope = 0 usually required for identifiability but simple centering helps stability)
        # For APC, we need 3 constraints. usually sum(k)=0, sum(g)=0, sum(linear_g)=0.
        # Let's just do sum(g)=0 for now.
        g_mean = np.mean(gamma_c[gamma_counts > 0])
        gamma_c[gamma_counts > 0] -= g_mean
        alpha_x += g_mean
        
    return alpha_x, kappa_t, gamma_c, years, ages, cohorts

def forecast_apc(alpha_x, kappa_t, gamma_c, years, ages, cohorts, n_forecast):
    # Forecast Kappa (Random Walk with Drift)
    drift = (kappa_t[-1] - kappa_t[0]) / (len(kappa_t) - 1)
    
    # Residuals for SE
    res_k = np.diff(kappa_t) - drift
    sigma_k = np.std(res_k, ddof=1)
    
    last_year = years[-1]
    forc_years = np.arange(last_year + 1, last_year + n_forecast + 1)
    
    # Forecast Gamma?
    # For the ages we care about (55, 65, 75) in the next 30 years:
    # Age 55 in 2050 -> Born 1995.
    # Max year in data is 2022.
    # 1995 is a cohort that exists in our data (observed at age 0 in 1995, age 27 in 2022).
    # So we have an estimate for gamma_1995.
    # We generally don't need to forecast gamma unless we look at very young ages in the future.
    # We will assume gamma is constant for existing cohorts (using the estimated value).
    # For future cohorts (not yet born), we would need to project, but we won't hit them for age 55+.
    
    # Prepare Forecast Dictionary
    forecasts = {}
    target_ages = [55, 65, 75]
    
    # Map ages to indices
    age_map = {age: i for i, age in enumerate(ages)}
    
    # Map cohorts to indices
    cohort_map = {c: i for i, c in enumerate(cohorts)}
    
    for age in target_ages:
        if age not in age_map:
            continue
        
        x_idx = age_map[age]
        ax = alpha_x[x_idx]
        
        age_forecast = {'point': [], 'lower': [], 'upper': []}
        
        last_kt = kappa_t[-1]
        
        for h in range(1, n_forecast + 1):
            year = last_year + h
            
            # 1. Forecast Kappa
            k_h = last_kt + h * drift
            se_k = sigma_k * np.sqrt(h)
            
            # 2. Get Cohort Effect
            cohort = year - age
            
            if cohort in cohort_map:
                c_idx = cohort_map[cohort]
                gc = gamma_c[c_idx]
            else:
                # Fallback for unknown cohorts (shouldn't happen for 55+)
                gc = 0 
            
            # 3. Combine: ln(m) = ax + kt + gc
            ln_m_point = ax + k_h + gc
            ln_m_lower = ax + (k_h - 1.96 * se_k) + gc
            ln_m_upper = ax + (k_h + 1.96 * se_k) + gc
            
            age_forecast['point'].append(np.exp(ln_m_point))
            age_forecast['lower'].append(np.exp(ln_m_lower))
            age_forecast['upper'].append(np.exp(ln_m_upper))
            
        forecasts[age] = age_forecast
        
    return forc_years, forecasts

def main():
    file_path = '12621-0001_de 2 with cohort.csv'
    print("Parsing data...")
    df = parse_german_data(file_path)
    
    if df.empty:
        print("Error: No data parsed.")
        return

    sexes = ['Male', 'Female']
    forecast_horizon = 2050 - 2022
    
    for sex in sexes:
        print(f"Fitting APC Model for {sex}...")
        alpha, kappa, gamma, years, ages, cohorts = fit_apc_model(df, sex)
        
        forc_years, forecasts = forecast_apc(alpha, kappa, gamma, years, ages, cohorts, forecast_horizon)
        
        # Plotting
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        target_ages = [55, 65, 75]
        
        subset = df[df['Sex'] == sex]
        
        for i, age in enumerate(target_ages):
            color = colors[i]
            
            # Historical
            hist_data = subset[subset['Age'] == age].sort_values('Year')
            plt.plot(hist_data['Year'], hist_data['Mortality Rate'], marker='o', color=color, label=f'Historical {age}')
            
            # Forecast
            if age in forecasts:
                f_data = forecasts[age]
                plt.plot(forc_years, f_data['point'], linestyle='--', marker='x', color=color, label=f'Forecast {age}')
                plt.fill_between(forc_years, f_data['lower'], f_data['upper'], color=color, alpha=0.2, label=f'95% CI {age}')
            
        plt.title(f'Mortality Rate Forecast for {sex}s (Age-Period-Cohort) with 95% CI')
        plt.xlabel('Year')
        plt.ylabel('Mortality Rate (q(x))')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_file = f'forecast_apc_{sex.lower()}.png'
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close()

if __name__ == "__main__":
    main()
