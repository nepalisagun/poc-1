import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def preprocess_powerball(data):
    try:
        # Convert 'draw_date' to datetime
        data['draw_date'] = pd.to_datetime(data['draw_date'])
        data['multiplier'] = pd.to_numeric(data['multiplier'])

        # Split 'winning_numbers' into separate columns
        winning_numbers_df = data['winning_numbers'].str.split(' ', expand=True)
        winning_numbers_df.columns = ['white_number_1', 'white_number_2', 'white_number_3', 'white_number_4', 'white_number_5', 'red_number']
        winning_numbers_df = winning_numbers_df.apply(pd.to_numeric)

        # Concatenate the original DataFrame with the new 'winning_numbers' DataFrame
        data = pd.concat([data, winning_numbers_df], axis=1)

        # Drop the original 'winning_numbers' column
        data.drop(columns=['winning_numbers'], inplace=True)

        # Extract features from 'draw_date' column
        data['Year'] = data['draw_date'].dt.year
        data['Month'] = data['draw_date'].dt.month
        data['Day'] = data['draw_date'].dt.day
        data['DayOfWeek'] = data['draw_date'].dt.dayofweek

        # Drop the original 'draw_date' column
        data.drop(columns=['draw_date'], inplace=True)

        # Drop duplicates if any
        data.drop_duplicates(inplace=True)

        # Handle missing values
        data.dropna(inplace=True)

        # Convert 'multiplier' column to integer
        data['multiplier'] = pd.to_numeric(data['multiplier'], errors='coerce')

        print("Preprocessed data after removing duplicates and handling missing values:")
        print(data.head())  # Print preprocessed data after handling duplicates and missing values

        # Split the data into features (X) and target (y)
        X = data.drop(columns=['red_number', 'white_number_1', 'white_number_2', 'white_number_3', 'white_number_4', 'white_number_5', 'multiplier'])  # Exclude 'multiplier' from input features
        y = data[['red_number', 'white_number_1', 'white_number_2', 'white_number_3', 'white_number_4', 'white_number_5', 'multiplier']]  # Include 'multiplier' in the target
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(X_train.dtypes)

        # Standardize features by removing the mean and scaling to unit variance
        scaler = StandardScaler()
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        numerical_features = X_train.select_dtypes(include=['int32', 'float64']).columns
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
        import traceback
        traceback.print_exc()  
        return None


if __name__ == "__main__":
    # Load your data
    data = pd.read_json("powerball.json")  # Replace with your actual data file

    # Call the preprocess_powerball function
    preprocessed_data = preprocess_powerball(data)

    # Print the preprocessed data
    print(preprocessed_data)
