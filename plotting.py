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
                            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    if not data:
        print("No data found matching criteria.")
        return

    df = pd.DataFrame(data)
    print("Extracted Data Head:")
    print(df.head())
    print("Unique Years Extracted:", df['Year'].unique())

    # Get unique combinations
    combinations = df[['Sex', 'Age']].drop_duplicates().sort_values(['Sex', 'Age'])

    for _, row in combinations.iterrows():
        sex = row['Sex']
        age = row['Age']
        subset = df[(df['Sex'] == sex) & (df['Age'] == age)].sort_values('Year')
        
        if subset.empty:
            continue

        plt.figure(figsize=(10, 6))
        plt.plot(subset['Year'], subset['Mortality Rate'], marker='o', linestyle='-')
        plt.title(f'Mortality Rate for {sex}s at {age} (1991-2022)')
        plt.xlabel('Year')
        plt.ylabel('Mortality Rate (q(x))')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(subset['Year'], rotation=45)
        plt.tight_layout()
        
        age_str = age.replace(' ', '_')
        output_file = f'mortality_rate_{sex.lower()}_{age_str}.png'
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        plt.close()

if __name__ == "__main__":
    main()
