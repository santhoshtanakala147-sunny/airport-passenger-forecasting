from lightgbm import LGBMRegressor

def create_model():
    return LGBMRegressor(
        n_estimators=200,
        random_state=42,
        verbose=-1
    )