import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Clean and preprocess the collected data.

    Args:
    - data: Dictionary containing fetched data from APIs.

    Returns:
    - preprocessed_data: Preprocessed and structured data ready for model training.
    """
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Drop duplicates if any
    df.drop_duplicates(inplace=True)

    # Handle missing values
    df.dropna(inplace=True)

    # Convert data types if necessary
    # (Not implemented in this example as it depends on the specific data)

    # Perform feature engineering if needed
    # (Not implemented in this example as it depends on the specific data)

    # Split the data into features (X) and target (y)
    X = df.drop(columns=['winning_numbers'])  # Assuming 'winning_numbers' is the target column
    y = df['winning_numbers']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    preprocessed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }

    return preprocessed_data

if __name__ == "__main__":
    # Assume collected_data is a dictionary containing fetched data from APIs
    # collected_data = {'API_1': data_1, 'API_2': data_2, ...}
    
    # Step 3: Preprocess the collected data
    preprocessed_data = preprocess_data(collected_data)

    # Display preprocessed data
    print("Preprocessed data:")
    for key, value in preprocessed_data.items():
        print(f"\n{key}:")
        print(value)
