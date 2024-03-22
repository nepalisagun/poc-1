from historical_powerball_data import read_powerball_data, read_megamillion_data
from preprocess_powerball_data import preprocess_powerball
from preprocess_megamillion_data import preprocess_megamillion
from train_powerball_models import train_model, evaluate_model
import joblib

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
        for key, X_train in preprocessed_powerball_data['X_train'].items():
            y_train = preprocessed_powerball_data['y_train'][key]
            model = train_model(X_train, y_train)
            trained_models[key] = model

        # Step 4: Evaluate Powerball models
        for key, model in trained_models.items():
            X_test = preprocessed_powerball_data['X_test'][key]
            y_test = preprocessed_powerball_data['y_test'][key]
            mse = evaluate_model(model, X_test, y_test)
            print(f"Model {key} Mean Squared Error: {mse}")

        # Step 5: Save trained models
        for key, model in trained_models.items():
            model_file = f"{key}_model.pkl"
            joblib.dump(model, model_file)
            print(f"Model {key} saved as {model_file}")
    else:
        print("Failed to preprocess Powerball data. Check for errors in the dataset or preprocessing steps.")
