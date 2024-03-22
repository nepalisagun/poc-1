import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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
    print("Training the model...")
    try:
        # Initialize the model
        model = RandomForestRegressor(random_state=42)

        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }

        # Initialize the grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5)

        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Get the best model from grid search
        best_grid_model = grid_search.best_estimator_

        # Perform cross-validation for the best grid model
        cv_scores = cross_val_score(best_grid_model, X_train, y_train, cv=5)
        print(f"Cross-validation scores for the best grid model: {cv_scores}")
        print(f"Mean cross-validation score for the best grid model: {np.mean(cv_scores)}")

        # Train the model with default parameters
        model.fit(X_train, y_train)
        print("Model with default parameters trained successfully.")

        # Perform cross-validation for the model with default parameters
        cv_scores_default = cross_val_score(model, X_train, y_train, cv=5)
        print(f"Cross-validation scores for the model with default parameters: {cv_scores_default}")
        print(f"Mean cross-validation score for the model with default parameters: {np.mean(cv_scores_default)}")

        # Compare the two models and choose the one with the higher mean cross-validation score
        if np.mean(cv_scores) > np.mean(cv_scores_default):
            print("The model from grid search has a higher mean cross-validation score and is chosen as the final model.")
            final_model = best_grid_model
        else:
            print("The model with default parameters has a higher mean cross-validation score and is chosen as the final model.")
            final_model = model

    except Exception as e:
        print(f"Error occurred during model training: {e}")
        return None

    return final_model

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
    print("Evaluating the model...")
    try:
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model evaluated successfully. Mean Squared Error: {mse}")
    except Exception as e:
        print(f"Error occurred during model evaluation: {e}")
        return None

    return mse

if __name__ == "__main__":
    # Assume preprocessed_data is a dictionary containing preprocessed data
    # preprocessed_data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

    # Step 4: Train models
    trained_models = {}
    for key, X_train in preprocessed_data['X_train'].items():
        y_train = preprocessed_data['y_train'][key]
        print(f"\nTraining model for {key}...")
        model = train_model(X_train, y_train)
        if model is not None:
            trained_models[key] = model
        else:
            print(f"Failed to train model for {key}. Skipping evaluation and saving steps.")

    # Step 5: Evaluate models
    for key, model in trained_models.items():
        X_test = preprocessed_data['X_test'][key]
        y_test = preprocessed_data['y_test'][key]
        print(f"\nEvaluating model for {key}...")
        mse = evaluate_model(model, X_test, y_test)
        if mse is None:
            print(f"Failed to evaluate model for {key}. Skipping saving step.")

    # Step 6: Save trained models
    for key, model in trained_models.items():
        model_file = f"{key}_model.pkl"
        try:
            joblib.dump(model, model_file)
            print(f"\nModel {key} saved as {model_file}")
        except Exception as e:
            print(f"Error occurred while saving model {key}: {e}")
