import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

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

def fit_cbd_model(df, sex):
    # Filter by sex
    subset = df[df['Sex'] == sex].copy()
    
    # Pivot to get Year x Age matrix
    # Ages are columns: 55 years, 65 years, 75 years
    pivot = subset.pivot(index='Year', columns='Age', values='Mortality Rate')
    pivot = pivot[['55 years', '65 years', '75 years']]
    
    years = pivot.index.values
    ages = np.array([55, 65, 75])
    mean_age = np.mean(ages) # 65.0
    
    # Transform to logit: ln(q / (1-q))
    Q = pivot.values
    logit_Q = np.log(Q / (1 - Q))
    
    # CBD Model: logit(q_{x,t}) = kappa_t^(1) + kappa_t^(2) * (x - x_bar)
    # We estimate kappa1 and kappa2 for each year using OLS (or simple line fitting since we have 3 points)
    
    kappa1_hist = []
    kappa2_hist = []
    
    # Regressor for the slope: (x - x_bar)
    x_minus_xbar = ages - mean_age
    
    for t in range(len(years)):
        y_t = logit_Q[t, :]
        # Fit linear regression: y = k1 + k2 * x_reg
        # np.polyfit(x, y, 1) returns [slope, intercept] -> [k2, k1]
        slope, intercept = np.polyfit(x_minus_xbar, y_t, 1)
        
        kappa1_hist.append(intercept)
        kappa2_hist.append(slope)
        
    return years, np.array(kappa1_hist), np.array(kappa2_hist), mean_age, ages

def forecast_cbd(years, kappa1, kappa2, mean_age, ages, n_forecast):
    # Multivariate Random Walk with Drift
    # K_t = K_{t-1} + drift + error
    
    # 1. Calculate Drifts
    drift1 = (kappa1[-1] - kappa1[0]) / (len(kappa1) - 1)
    drift2 = (kappa2[-1] - kappa2[0]) / (len(kappa2) - 1)
    
    # 2. Calculate Residuals and Covariance Matrix
    res1 = np.diff(kappa1) - drift1
    res2 = np.diff(kappa2) - drift2
    
    # Covariance matrix of residuals
    cov_matrix = np.cov(np.vstack([res1, res2]), ddof=1)
    var1 = cov_matrix[0, 0]
    var2 = cov_matrix[1, 1]
    cov12 = cov_matrix[0, 1]
    
    # 3. Forecast
    last_year = years[-1]
    forc_years = np.arange(last_year + 1, last_year + n_forecast + 1)
    
    forecasts = {}
    
    for i, age in enumerate(ages):
        x_diff = age - mean_age
        
        age_forecast = {'point': [], 'lower': [], 'upper': []}
        
        last_k1 = kappa1[-1]
        last_k2 = kappa2[-1]
        
        for h in range(1, n_forecast + 1):
            # Forecast Kappas
            k1_h = last_k1 + h * drift1
            k2_h = last_k2 + h * drift2
            
            # Forecast Logit
            logit_h = k1_h + k2_h * x_diff
            
            # Variance of Logit Forecast
            # Var(logit) = Var(k1) + (x-x_bar)^2 * Var(k2) + 2*(x-x_bar)*Cov(k1, k2)
            # Var(k_t+h) = h * Var(error)
            
            var_logit_h = h * (var1 + (x_diff**2) * var2 + 2 * x_diff * cov12)
            se_logit_h = np.sqrt(var_logit_h)
            
            # Calculate CI in Logit scale
            logit_lower = logit_h - 1.96 * se_logit_h
            logit_upper = logit_h + 1.96 * se_logit_h
            
            # Transform back to q
            # q = exp(logit) / (1 + exp(logit))
            def logit_to_q(l):
                return np.exp(l) / (1 + np.exp(l))
            
            age_forecast['point'].append(logit_to_q(logit_h))
            age_forecast['lower'].append(logit_to_q(logit_lower))
            age_forecast['upper'].append(logit_to_q(logit_upper))
            
        forecasts[age] = age_forecast
        
    return forc_years, forecasts

def main():
    file_path = '12621-0001_en.csv'
    df = parse_data(file_path)
    
    sexes = ['Male', 'Female']
    ages_labels = {55: '55 years', 65: '65 years', 75: '75 years'}
    
    forecast_horizon = 2050 - 2022
    
    for sex in sexes:
        print(f"Fitting CBD Model for {sex}...")
        years, k1, k2, mean_age, ages_num = fit_cbd_model(df, sex)
        
        forc_years, forecasts = forecast_cbd(years, k1, k2, mean_age, ages_num, forecast_horizon)
        
        # Plotting
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Get historical data for plotting
        subset = df[df['Sex'] == sex]
        
        for i, age_num in enumerate(ages_num):
            age_label = ages_labels[age_num]
            color = colors[i]
            
            # Historical
            hist_data = subset[subset['Age'] == age_label].sort_values('Year')
            plt.plot(hist_data['Year'], hist_data['Mortality Rate'], marker='o', color=color, label=f'Historical {age_label}')
            
            # Forecast
            f_data = forecasts[age_num]
            plt.plot(forc_years, f_data['point'], linestyle='--', marker='x', color=color, label=f'Forecast {age_label}')
            plt.fill_between(forc_years, f_data['lower'], f_data['upper'], color=color, alpha=0.2, label=f'95% CI {age_label}')
            
        plt.title(f'Mortality Rate Forecast for {sex}s (Cairns-Blake-Dowd) with 95% CI')
        plt.xlabel('Year')
        plt.ylabel('Mortality Rate (q(x))')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        output_file = f'forecast_cbd_{sex.lower()}.png'
        plt.savefig(output_file)
        print(f"Saved plot to {output_file}")
        plt.close()

if __name__ == "__main__":
    main()
