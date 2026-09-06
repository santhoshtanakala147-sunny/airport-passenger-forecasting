# ✈️ Airport Passenger Traffic Forecasting

Machine-learning based forecasting of passenger traffic for **domestic (national)** and **international** Hyderabad airport routes using DGCA/OpenCity operational data.

## What this project does
- Cleans and preprocesses aviation traffic data.
- Engineers passenger/freight ratios, seasonality and lag features.
- Compares Ridge, Random Forest, Gradient Boosting, XGBoost and LightGBM.
- Uses time-based train/validation/test evaluation.
- Tunes the selected model with GridSearchCV.
- Produces route-level future passenger forecasts.
- Saves trained models, Excel predictions and evaluation plots.

## Repository structure
```text
airport-passenger-forecasting/
├── data/
│   ├── domestic/          # domestic DGCA route data
│   └── international/     # international route data
├── src/
│   ├── domestic_forecast.py
│   ├── international_forecast.py
│   └── dsa_utils.py
├── models/                # saved trained models
├── outputs/               # predictions, metrics and plots
├── docs/                  # supporting project result document
├── requirements.txt
└── README.md
```

## Run
```bash
pip install -r requirements.txt
python src/domestic_forecast.py
python src/international_forecast.py
```

> Run the scripts from the repository root. The international script expects `data/international/international.xlsx`; the repository copy is already provided.

## International result
The supplied international experiment uses quarterly route-level data covering 2015–2025, with 48 unique routes. The final experiment and prediction outputs are included under `outputs/` and the trained model under `models/`.

## Domestic result
The domestic pipeline uses monthly route-level data, engineered lag/seasonality features, model comparison and tuned regression for future route forecasts.

## Notes
The repository intentionally keeps the datasets and generated result files that were supplied for the project so the work is reproducible and easy to demonstrate.
