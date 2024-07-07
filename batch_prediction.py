import os
import json
import requests
import pandas as pd

# Set the directory containing the JSON files
PREDICTION_DATA_DIR = "data/cs-production"
PREDICTION_URL = "http://127.0.0.1:8000/predict"

def load_json_files(directory):
    """Load JSON files from the specified directory."""
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    return json_files

def preprocess_data(transactions):
    """Preprocess the list of transactions and convert to the required format."""
    df = pd.DataFrame(transactions)
    # Assuming that you need to convert this DataFrame to the required format
    country = df['country'].iloc[0]
    date = df['year'].iloc[0] + "-" + df['month'].iloc[0] + "-" + df['day'].iloc[0]
    # Perform necessary aggregations or transformations here

    # Create a dictionary for the prediction endpoint
    prediction_data = {
        "country": country,
        "year": df['year'].iloc[0],
        "month": df['month'].iloc[0],
        "day": df['day'].iloc[0],
        "test": True  # Change to False if this is not a test
    }
    return prediction_data

def make_prediction(data):
    """Make a prediction by sending a POST request to the FastAPI endpoint."""
    response = requests.post(PREDICTION_URL, json=data)
    return response.json()

def batch_predict(directory):
    """Run predictions on all JSON files in the specified directory."""
    json_files = load_json_files(directory)
    results = []

    for json_file in json_files:
        file_path = os.path.join(directory, json_file)
        print(f"Processing {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                transactions = json.load(f)
            prediction_data = preprocess_data(transactions)
            result = make_prediction(prediction_data)
            print(f"Result for {json_file}: {result}")
            results.append({json_file: result})
        except Exception as e:
            print(f"Failed to process {json_file}: {e}")
            results.append({json_file: str(e)})
    
    return results

if __name__ == "__main__":
    results = batch_predict(PREDICTION_DATA_DIR)
    # Save results to a file
    with open("batch_prediction_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Batch prediction complete. Results saved to batch_prediction_results.json.")