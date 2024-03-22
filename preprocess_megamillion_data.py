import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_megamillion(data):
    """
    Clean and preprocess the collected Mega Millions data.

    Args:
    - data: DataFrame containing fetched Mega Millions data from CSV.

    Returns:
    - preprocessed_data: Preprocessed and structured Mega Millions data ready for model training.
    """
    # Convert 'Draw Date' column to datetime format
    data['Draw Date'] = pd.to_datetime(data['Draw Date'], format='%m/%d/%Y')

    # Split 'Winning Numbers' column into individual numbers
    data['Winning Numbers'] = data['Winning Numbers'].str.split()

    # Convert 'Multiplier' column to integer type and handle missing values
    data['Multiplier'] = data['Multiplier'].fillna(1).astype(int)

    # Standardize 'Mega Ball' column
    scaler = StandardScaler()
    data['Mega Ball'] = scaler.fit_transform(data['Mega Ball'].values.reshape(-1, 1))

    # Split data into features (X) and target (y)
    X = data[['Mega Ball', 'Multiplier']]  # Features
    y = data['Winning Numbers']  # Target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features using StandardScaler
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
