# import requests
# import pandas as pd

# # Define function to fetch data from API
# def fetch_data(api_url):
#     try:
#         response = requests.get(api_url)
#         response.raise_for_status()  # Raise an exception for HTTP errors
#         data = response.json()
#         return data
#     except Exception as e:
#         print(f"Failed to fetch data from {api_url}: {e}")
#         return None

# # Define function to collect data from multiple APIs
# def collect_data(api_urls):
#     collected_data = {}
#     for index, api_url in enumerate(api_urls, start=1):
#         print(f"Fetching data from API {index}/{len(api_urls)}...")
#         data = fetch_data(api_url)
#         if data:
#             collected_data[f"API_{index}"] = data
#     return collected_data

# if __name__ == "__main__":
#     # Define URLs of API endpoints for fetching historical Powerball draw data
#     api_urls = [
#         "https://www.lotteryresults.io/api/powerball",
#         "https://www.lottodata.online/powerball/draws",
#         "https://app.swaggerhub.com/apis/lotto649/ca-lottery/1.0.0/powerball/draws",
#         "https://www.thelotter.com/powerball/numbers-archive/"
#     ]

#     # Step 1: Collect data from APIs
#     collected_data = collect_data(api_urls)

#     # Step 2: Preprocess the collected data (not implemented in this script)

#     # Display the collected data
#     print("\nCollected data:")
#     for key, value in collected_data.items():
#         print(f"\nAPI: {key}")
#         print(value)


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
