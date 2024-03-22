from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(X_train, y_train):
    """
    Train a predictive model using RandomForestRegressor.

    Args:
    - X_train: Training features.
    - y_train: Training target.

    Returns:
    - trained_model: Trained predictive model.
    """
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using mean squared error (MSE).

    Args:
    - model: Trained predictive model.
    - X_test: Testing features.
    - y_test: Testing target.

    Returns:
    - mse: Mean squared error.
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return mse

if __name__ == "__main__":
    # Assume preprocessed_data is a dictionary containing preprocessed data
    # preprocessed_data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    # Step 4: Train models
    trained_models = {}
    for key, X_train in preprocessed_data['X_train'].items():
        y_train = preprocessed_data['y_train'][key]
        model = train_model(X_train, y_train)
        trained_models[key] = model

    # Step 5: Evaluate models
    for key, model in trained_models.items():
        X_test = preprocessed_data['X_test'][key]
        y_test = preprocessed_data['y_test'][key]
        mse = evaluate_model(model, X_test, y_test)
        print(f"Model {key} Mean Squared Error: {mse}")

    # Step 6: Save trained models
    for key, model in trained_models.items():
        model_file = f"{key}_model.pkl"
        joblib.dump(model, model_file)
        print(f"Model {key} saved as {model_file}")
