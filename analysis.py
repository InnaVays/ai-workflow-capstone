import os
import json
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = "logs"
PREDICT_LOG_FILE = "predict_log.json"

def load_logs(log_file):
    log_path = os.path.join(LOG_DIR, log_file)
    if not os.path.exists(log_path):
        raise Exception(f"Log file {log_file} not found")

    with open(log_path, "r") as file:
        logs = json.load(file)
    
    return logs

def analyze_relationship(logs):
    # Convert logs to DataFrame
    df = pd.DataFrame(logs)
    
    # Ensure business_metric is a dictionary and extract metrics
    def extract_metric(metric, key):
        if isinstance(metric, dict) and key in metric:
            return metric[key]
        return None

    df['absolute_error'] = df['business_metric'].apply(lambda x: extract_metric(x, 'absolute_error'))
    df['mse'] = df['business_metric'].apply(lambda x: extract_metric(x, 'mse'))
    df['r2_score'] = df['business_metric'].apply(lambda x: extract_metric(x, 'r2_score'))

    # Drop rows with None values
    df.dropna(subset=['absolute_error', 'mse', 'r2_score'], inplace=True)

    # Calculate correlation between model performance metrics and business metric
    correlation = df[['absolute_error', 'mse', 'r2_score']].corr()

    print("Correlation matrix:")
    print(correlation)

    # Plot relationships
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(df['absolute_error'], df['r2_score'], alpha=0.5)
    plt.title("Absolute Error vs R^2 Score")
    plt.xlabel("Absolute Error")
    plt.ylabel("R^2 Score")

    plt.subplot(1, 3, 2)
    plt.scatter(df['mse'], df['r2_score'], alpha=0.5)
    plt.title("MSE vs R^2 Score")
    plt.xlabel("MSE")
    plt.ylabel("R^2 Score")

    plt.subplot(1, 3, 3)
    plt.scatter(df['absolute_error'], df['mse'], alpha=0.5)
    plt.title("Absolute Error vs MSE")
    plt.xlabel("Absolute Error")
    plt.ylabel("MSE")

    plt.tight_layout()
    plt.savefig("business_metrics_analysis.png")
    plt.show()

if __name__ == "__main__":
    logs = load_logs(PREDICT_LOG_FILE)
    analyze_relationship(logs)