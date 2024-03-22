import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_powerball(data):
    try:
        print("Original data:")
        print(data.head())  # Print original data before preprocessing

        # Drop duplicates if any
        data.drop_duplicates(inplace=True)

        # Handle missing values
        data.dropna(inplace=True)

        # Extract date part from 'draw_date' column
        data['Draw Date'] = pd.to_datetime(data['draw_date'].str.split('T').str[0], errors='coerce')

        # Drop rows with invalid dates
        data.dropna(subset=['Draw Date'], inplace=True)

        # Drop the original 'draw_date' column
        data.drop(columns=['draw_date'], inplace=True)

        # Split winning_numbers into list of integers
        data['winning_numbers'] = data['winning_numbers'].str.split().apply(lambda x: [int(num) for num in x])

        # Convert 'multiplier' column to integer
        data['multiplier'] = pd.to_numeric(data['multiplier'], errors='coerce')

        print("Preprocessed data after removing duplicates and handling missing values:")
        print(data.head())  # Print preprocessed data after handling duplicates and missing values

        # Split the data into features (X) and target (y)
        X = data.drop(columns=['winning_numbers'])  # Assuming 'winning_numbers' is the target column
        y = data['winning_numbers']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = scaler.transform(X_test[numerical_features])

        preprocessed_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        return preprocessed_data
    except Exception as e:
        print(f"Error occurred during preprocessing: {e}")
        return None

if __name__ == "__main__":
    # Load Powerball data from a JSON file
    powerball_data = pd.read_json("powerball.json")

    # Step 2: Preprocess the collected Powerball data
    preprocessed_powerball_data = preprocess_powerball(powerball_data)

    if preprocessed_powerball_data:
        # Display preprocessed data
        print("Preprocessed Powerball data:")
        for key, value in preprocessed_powerball_data.items():
            print(f"\n{key}:")
            print(value)
    else:
        print("Failed to preprocess Powerball data. Check for errors in the dataset or preprocessing steps.")
