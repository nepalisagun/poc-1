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
    print("\nAfter converting 'Draw Date' column to datetime format:")
    print(data.head())

    # Split 'Winning Numbers' column into individual numbers
    data['Winning Numbers'] = data['Winning Numbers'].str.split()
    print("\nAfter splitting 'Winning Numbers' column into individual numbers:")
    print(data.head())

    # Convert 'Multiplier' column to integer type and handle missing values
    data['Multiplier'] = data['Multiplier'].fillna(1).astype(int)
    print("\nAfter converting 'Multiplier' column to integer type and handling missing values:")
    print(data.head())

    # Standardize 'Mega Ball' column
    scaler = StandardScaler()
    data['Mega Ball'] = scaler.fit_transform(data['Mega Ball'].values.reshape(-1, 1))
    print("\nAfter standardizing 'Mega Ball' column:")
    print(data.head())

    # Split data into features (X) and target (y)
    X = data[['Mega Ball', 'Multiplier']]  # Features
    y = data['Winning Numbers']  # Target
    print("\nFeatures (X) and target (y):")
    print("X:")
    print(X.head())
    print("y:")
    print(y.head())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("\nAfter splitting data into training and testing sets:")
    print("X_train:")
    print(X_train.head())
    print("X_test:")
    print(X_test.head())
    print("y_train:")
    print(y_train.head())
    print("y_test:")
    print(y_test.head())

    # Standardize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\nAfter standardizing features using StandardScaler:")
    print("X_train_scaled:")
    print(X_train_scaled[:5])  # Print first 5 rows
    print("X_test_scaled:")
    print(X_test_scaled[:5])  # Print first 5 rows

    preprocessed_data = {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test
    }

    return preprocessed_data