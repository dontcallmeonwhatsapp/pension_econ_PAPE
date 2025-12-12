import pandas as pd
import matplotlib.pyplot as plt
import re

def main():
    file_path = '12621-0001_en.csv'
    data = []
    current_year = None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(';')
                if not parts:
                    continue
                
                # Check for year line (e.g., "1991/93")
                # We take the first 4 digits as the year
                if re.match(r'^\d{4}/\d{2}$', parts[0]):
                    current_year = int(parts[0][:4])
                    continue
                
                if current_year is not None:
                    # Check for Male, 55 years
                    # parts[0] == 'Male', parts[1] == '55 years'
                    # Ensure we have enough parts
                    if len(parts) > 2 and parts[0] == 'Male' and parts[1] == '55 years':
                        try:
                            # The file uses '.' as decimal separator based on inspection
                            rate = float(parts[2])
                            data.append({'Year': current_year, 'Mortality Rate': rate})
                        except ValueError:
                            continue
                            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    if not data:
        print("No data found matching criteria.")
        return

    df = pd.DataFrame(data)
    print("Extracted Data:")
    print(df)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(df['Year'], df['Mortality Rate'], marker='o', linestyle='-', color='b')
    plt.title('Mortality Rate for Males at Age 55 (1991-2022)')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate (q(x))')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(df['Year'], rotation=45)
    plt.tight_layout()
    
    output_file = 'mortality_rate_male_55.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()
