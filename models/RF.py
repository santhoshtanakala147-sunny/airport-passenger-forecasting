from sklearn.ensemble import RandomForestRegressor

def create_model():
    return RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )