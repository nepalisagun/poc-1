# from tkinter import *
# from tkinter import ttk
# import joblib

# def load_model(model_file):
#     """
#     Load a trained predictive model from disk.

#     Args:
#     - model_file: File path of the saved model.

#     Returns:
#     - model: Loaded predictive model.
#     """
#     try:
#         model = joblib.load(model_file)
#         return model
#     except Exception as e:
#         print(f"Failed to load model from {model_file}: {e}")
#         return None

# def predict_powerball_numbers(model_file, features):
#     """
#     Generate predictions for the next set of Powerball numbers using the trained model.

#     Args:
#     - model_file: File path of the trained model.
#     - features: Input features for prediction.

#     Returns:
#     - predictions: Predicted Powerball numbers.
#     """
#     # Load the trained model
#     model = load_model(model_file)
#     if model is None:
#         return None

#     # Generate predictions
#     predictions = model.predict(features)

#     return predictions

# def predict_powerball():
#     # Load the trained model
#     model_file = "powerball_model.pkl"  # File path of the trained Powerball model

#     # Placeholder for input features
#     features = []  # Add actual input features

#     # Predict Powerball numbers
#     predictions = predict_powerball_numbers(model_file, features)

#     # Display predicted numbers
#     predicted_numbers_label.config(text=f"Predicted Powerball Numbers: {predictions}")

# # Create main window
# root = Tk()
# root.title("Powerball Prediction")

# # Create frame
# frame = Frame(root)
# frame.pack(padx=10, pady=10)

# # Create label and combobox for selecting lottery
# lottery_label = Label(frame, text="Select Lottery:")
# lottery_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

# lottery_combobox = ttk.Combobox(frame, values=["Powerball", "Mega Millions"])  # Add other lotteries if needed
# lottery_combobox.grid(row=0, column=1, padx=5, pady=5)

# # Create button for initiating predictions
# predict_button = Button(frame, text="Predict", command=predict_powerball)
# predict_button.grid(row=1, column=0, columnspan=2, pady=10)

# # Create label for displaying predicted numbers
# predicted_numbers_label = Label(frame, text="")
# predicted_numbers_label.grid(row=2, column=0, columnspan=2, pady=5)

# # Run the application
# root.mainloop()


import joblib
from sklearn.ensemble import RandomForestRegressor
from tkinter import *
from tkinter import ttk  # Assuming you're using ttk for widgets

def train_model(X_train, y_train):
    """
    Train a predictive model using RandomForestRegressor.

    Args:
        X_train (2D array): Training features.
        y_train (1D array): Training target.

    Returns:
        trained_model: Trained predictive model.
    """

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def save_model(model, model_file):
    """
    Save the trained model to a file.

    Args:
        model: Trained predictive model.
        model_file (str): File path for saving the model.
    """

    try:
        joblib.dump(model, model_file)
        print(f"Model saved as {model_file}")
    except Exception as e:
        print(f"Failed to save model to {model_file}: {e}")

def train_and_save_model():
    # Replace these placeholders with your actual data loading and preprocessing logic
    # Ensure X_train is a 2D array and y_train is a 1D array
    X_train = ...  # Load or prepare your training features (2D array)
    y_train = ...  # Load or prepare your training target (1D array)

    # Check for empty arrays and reshape if necessary
    if len(X_train) == 0 or len(y_train) == 0:
        print("Error: Training data is empty. Please provide data for training.")
        return

    if X_train.ndim == 1:
        # Reshape if it's a single feature
        X_train = X_train.reshape(-1, 1)
    elif X_train.ndim > 2:
        print("Error: Training features should be a 2D array. Please check your data loading process.")
        return

    # Train the model
    model = train_model(X_train, y_train)

    # Save the trained model to a file
    model_file = "powerball_model.pkl"
    save_model(model, model_file)

    # Provide feedback to the user
    train_feedback_label.config(text="Model trained and saved successfully!")

# Create main window (assuming your GUI code remains similar)
root = Tk()
root.title("Powerball Prediction")

# Create frame
frame = Frame(root)
frame.pack(padx=10, pady=10)

# Create button for initiating model training
train_button = Button(frame, text="Train Model", command=train_and_save_model)
train_button.grid(row=0, column=0, columnspan=2, pady=10)

# Create label for displaying training feedback
train_feedback_label = Label(frame, text="")
train_feedback_label.grid(row=1, column=0, columnspan=2, pady=5)

# Run the application
root.mainloop()
