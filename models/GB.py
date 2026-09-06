from sklearn.ensemble import GradientBoostingRegressor

def create_model():
    return GradientBoostingRegressor(
        n_estimators=200,
        random_state=42
    )