# Mortality Projection for Germany

This project forecasts mortality rates by gender for Germany using three stochastic mortality models, as part of the **Population Ageing and Pension Economics** course.

📄 **Task**: [PAPE Project.pdf](PAPE%20Project.pdf)  
📄 **Report**: [PAPE.pdf](PAPE.pdf)

## Models Implemented

| Model                   | File                    | Description                               |
| ----------------------- | ----------------------- | ----------------------------------------- |
| Lee-Carter              | `forecast_mortality.py` | Classic model with age and period effects |
| Age-Period-Cohort (APC) | `forecast_apc.py`       | Adds cohort effects                       |
| Cairns-Blake-Dowd (CBD) | `forecast_cbd.py`       | Two-factor model for older ages           |
| Plat                    | `forecast_plat.py`      | Hybrid combining LC and CBD features      |

## Setup

```bash
pip install pandas numpy matplotlib scipy plotly
```

## Usage

Run any forecast script:

```bash
python forecast_mortality.py   # Lee-Carter
python forecast_apc.py         # APC model
python forecast_cbd.py         # CBD model
python forecast_plat.py        # Plat model
```

Generate life expectancy forecasts:

```bash
python plot_life_expectancy.py
```

Plot historical mortality rates:

```bash
python plotting.py
```

## Output

Plots are saved to `Forecast Plots/` directory.

## Data

Source: [Statistisches Bundesamt (Destatis)](https://www.destatis.de)

- `12621-0001_en.csv` — German mortality data (English)
- `12621-0001_de 2 with cohort.csv` — German mortality data with cohort info (German)
