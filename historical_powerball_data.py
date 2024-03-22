import json
import pandas as pd

# Define function to read Powerball data from JSON file
def read_powerball_data(json_file):
    try:
        with open(json_file, 'r') as file:
            powerball_data = json.load(file)
            # Convert list of dictionaries to DataFrame
            powerball_data = pd.DataFrame(powerball_data)
            return powerball_data
    except Exception as e:
        print(f"Failed to read Powerball data from {json_file}: {e}")
        return None

# Define function to read Mega Millions data from CSV file
def read_megamillion_data(csv_file):
    try:
        megamillion_data = pd.read_csv(csv_file)
        return megamillion_data
    except Exception as e:
        print(f"Failed to read Mega Millions data from {csv_file}: {e}")
        return None

if __name__ == "__main__":
    # Step 1: Read data from local files
    powerball_data = read_powerball_data("powerball.json")
    megamillion_data = read_megamillion_data("megam.csv")

    # Step 2: Preprocess the collected data (not implemented in this script)

    # Display the collected data
    print("\nPowerball data:")
    print(powerball_data)

    print("\nMega Millions data:")
    print(megamillion_data)
