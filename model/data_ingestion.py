import os
import json
import pandas as pd

def load_json_files(data_dir):

    all_data = []
    json_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.json')]

    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)

    df = pd.DataFrame(all_data)
    return df

def save_to_csv(df, output_path):

    df.to_csv(output_path, index=False)

def main(data_dir, output_path):

    df = load_json_files(data_dir)
    save_to_csv(df, output_path)
    print(f"Data ingested and saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data_dir = "data/cs-production"
    output_path = "batch_data.csv"
    main(data_dir, output_path)
