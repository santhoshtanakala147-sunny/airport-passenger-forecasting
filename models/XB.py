from xgboost import XGBRegressor

def create_model():
    return XGBRegressor(
        n_estimators=200,
        random_state=42,
        verbosity=0
    )