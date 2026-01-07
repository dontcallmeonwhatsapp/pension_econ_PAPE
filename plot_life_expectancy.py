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
    subset = df[df['Sex'] == sex].copy()
    pivot = subset.pivot(index='Year', columns='Age', values='Mortality Rate')
    pivot = pivot[['55 years', '65 years', '75 years']]
    log_rates = np.log(pivot.values)
    years = pivot.index.tolist()
    return log_rates, years

def lee_carter_fit(log_rates):
    """
    Fit Lee-Carter model: ln(m_x,t) = a_x + b_x * k_t
    Returns ax, bx, kt
    """
    log_rates = np.array(log_rates)
    n_years, n_ages = log_rates.shape
    
    # ax = mean over time for each age
    ax = np.mean(log_rates, axis=0)
    
    # Center the data
    centered = log_rates - ax
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # First component
    bx = Vt[0, :]
    kt = U[:, 0] * S[0]
    
    # Normalize bx to sum to 1
    bx_sum = np.sum(bx)
    bx = bx / bx_sum
    kt = kt * bx_sum
    
    return ax, bx, kt

def calculate_life_expectancy(qx_dict, start_age, max_age=100):
    """
    Calculate period life expectancy at a given age using mortality rates.
    """
    ex = 0.0
    px_cumulative = 1.0
    
    for age in range(start_age, max_age):
        if age in qx_dict:
            qx = qx_dict[age]
        else:
            known_ages = sorted(qx_dict.keys())
            if age < known_ages[0]:
                qx = qx_dict[known_ages[0]]
            elif age > known_ages[-1]:
                qx = min(qx_dict[known_ages[-1]] * 1.1, 0.99)
            else:
                lower_age = max([a for a in known_ages if a <= age])
                upper_age = min([a for a in known_ages if a >= age])
                if lower_age == upper_age:
                    qx = qx_dict[lower_age]
                else:
                    t = (age - lower_age) / (upper_age - lower_age)
                    qx = qx_dict[lower_age] * (1 - t) + qx_dict[upper_age] * t
        
        qx = max(0.0001, min(qx, 0.9999))
        ex += px_cumulative * (1 - qx / 2)
        px_cumulative *= (1 - qx)
        
        if px_cumulative < 0.0001:
            break
    
    return ex

def main():
    file_path = '12621-0001_en.csv'
    df = parse_data(file_path)
    
    age_nums = [55, 65, 75]
    sexes = ['Male', 'Female']
    
    results = {sex: {'years': [], 'e65': [], 'e75': []} for sex in sexes}
    
    for sex in sexes:
        print(f"Calculating life expectancy for {sex}...")
        log_rates, years = prepare_lee_carter_data(df, sex)
        
        last_year = years[-1]
        
        # Fit Lee-Carter model
        ax, bx, kt = lee_carter_fit(log_rates)
        
        # Calculate drift
        c = (kt[-1] - kt[0]) / (len(kt) - 1)
        
        # Forecast for each year from 2023 to 2050
        forecast_years = list(range(last_year + 1, 2051))
        
        for year in forecast_years:
            h = year - last_year
            k_h = kt[-1] + h * c
            
            # Calculate mortality rates for this year
            qx_dict = {}
            for i, age in enumerate(age_nums):
                ln_mx = ax[i] + bx[i] * k_h
                qx_dict[age] = np.exp(ln_mx)
            
            # Calculate life expectancy at 65 and 75
            e65 = calculate_life_expectancy(qx_dict, 65)
            e75 = calculate_life_expectancy(qx_dict, 75)
            
            results[sex]['years'].append(year)
            results[sex]['e65'].append(e65)
            results[sex]['e75'].append(e75)
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'Male': '#2E86AB', 'Female': '#A23B72'}
    
    # Plot Life Expectancy at Age 65
    ax1 = axes[0]
    for sex in sexes:
        ax1.plot(results[sex]['years'], results[sex]['e65'], 
                 marker='o', markersize=3, linewidth=2, 
                 color=colors[sex], label=sex)
    
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Life Expectancy (years)', fontsize=12)
    ax1.set_title('Period Life Expectancy at Age 65\n(Lee-Carter Forecast)', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(2023, 2050)
    
    # Plot Life Expectancy at Age 75
    ax2 = axes[1]
    for sex in sexes:
        ax2.plot(results[sex]['years'], results[sex]['e75'], 
                 marker='o', markersize=3, linewidth=2, 
                 color=colors[sex], label=sex)
    
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Life Expectancy (years)', fontsize=12)
    ax2.set_title('Period Life Expectancy at Age 75\n(Lee-Carter Forecast)', fontsize=14)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(2023, 2050)
    
    plt.tight_layout()
    
    output_file = 'Forecast Plots/life_expectancy_forecast.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*60)
    print("Life Expectancy Summary (Lee-Carter Model)")
    print("="*60)
    print(f"{'Year':<8} {'Male e65':<12} {'Male e75':<12} {'Female e65':<12} {'Female e75':<12}")
    print("-"*60)
    
    key_years = [2023, 2030, 2040, 2050]
    for year in key_years:
        idx = results['Male']['years'].index(year)
        print(f"{year:<8} {results['Male']['e65'][idx]:<12.1f} {results['Male']['e75'][idx]:<12.1f} "
              f"{results['Female']['e65'][idx]:<12.1f} {results['Female']['e75'][idx]:<12.1f}")

if __name__ == "__main__":
    main()
