from historical_powerball_data import read_powerball_data, read_megamillion_data
from preprocess_powerball_data import preprocess_powerball
from preprocess_megamillion_data import preprocess_megamillion
from predict_powerball_numbers import load_model, predict_powerball_numbers
from train_powerball_models import train_model, evaluate_model
from evaluate_powerball_models import evaluate_model_performance, plot_predicted_numbers
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
def evaluate_model_performance(y_true, y_pred):
    """
    Evaluate the performance of a predictive model using mean squared error (MSE).

    Args:
    - y_true: True target values.
    - y_pred: Predicted target values.

    Returns:
    - mse: Mean squared error.
    """
    mse = mean_squared_error(y_true, y_pred)

    return mse


if __name__ == "__main__":
    # Step 1: Read data from local files
    powerball_data = read_powerball_data("powerball.json")
    megamillion_data = read_megamillion_data("megam.csv")

    # Step 2: Preprocess the collected data
    preprocessed_powerball_data = preprocess_powerball(powerball_data)
    preprocessed_megamillion_data = preprocess_megamillion(megamillion_data)

    # Display the preprocessed data
    print("\nPreprocessed Powerball data:")
    print(preprocessed_powerball_data)

    print("\nPreprocessed Mega Millions data:")
    print(preprocessed_megamillion_data)

    # Step 3: Train Powerball models
    trained_models = {}
    if preprocessed_powerball_data:
        X_train = preprocessed_powerball_data['X_train']
        y_train = preprocessed_powerball_data['y_train']
        model = train_model(X_train, y_train)
        trained_models['powerball'] = model

        # Step 4: Evaluate Powerball models
        X_test = preprocessed_powerball_data['X_test']
        y_test = preprocessed_powerball_data['y_test']
        mse = evaluate_model(model, X_test, y_test)
        print(f"Model Mean Squared Error: {mse}")

        # Evaluate model performance using mean squared error
        y_pred = model.predict(X_test)
        mse = evaluate_model_performance(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Step 5: Save trained models
        model_file = "powerball_model.pkl"
        joblib.dump(model, model_file)
        print(f"Model saved as {model_file}")
    else:
        print("Failed to preprocess Lottery data. Check for errors in the dataset or preprocessing steps.")

    # Step 7: Predict Powerball numbers
    model_file = "powerball_model.pkl"  # File path of the trained Powerball model
    user_input = {
    'features': [2024, 3, 23, 3]  # Example features: [Year, Month, Day, DayOfWeek]
    }

    features = np.array(user_input['features']).reshape(1, -1)
    predictions = predict_powerball_numbers(model_file, features)

    # Display the predicted Powerball numbers
    print("Predicted Powerball numbers:")
    print(predictions)

    # Plot the predicted Powerball numbers
    plot_predicted_numbers(predictions)
