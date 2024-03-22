from main_script import preprocess_powerball, train_model, evaluate_model
import joblib
from sklearn.ensemble import RandomForestRegressor
from tkinter import *
from tkinter import ttk  # Assuming you're using ttk for widgets

# ... rest of your powerball_prediction_ui.py code ...

def train_and_save_model():
    # Load or prepare your training features (2D array) and target (1D array)
    powerball_data = read_powerball_data("powerball.json")  # Assuming you have a function to read the data
    preprocessed_data = preprocess_powerball(powerball_data)
    X_train = preprocessed_data['X_train']
    y_train = preprocessed_data['y_train']

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
