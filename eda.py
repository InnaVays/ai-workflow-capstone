import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the data directory exists
DATA_DIR = "data/cs-train"

def load_data(data_dir):
    """Load data from JSON files into a single DataFrame."""
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    all_data = []
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        with open(file_path, 'r') as f:
            data = pd.read_json(f)
            all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

def plot_time_series(df, country=None):
    """Plot time-series data."""
    if country:
        df = df[df['country'] == country]
    df['invoice_date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.groupby('invoice_date').agg({'price': 'sum'}).reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(df['invoice_date'], df['price'])
    plt.title(f'Time Series of Revenue{" for " + country if country else ""}')
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.grid(True)
    plt.show()

def plot_revenue_distribution(df):
    """Plot distribution of revenue."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price'], bins=50, kde=True)
    plt.title('Distribution of Revenue')
    plt.xlabel('Revenue')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_views_distribution(df):
    """Plot distribution of times viewed."""
    plt.figure(figsize=(12, 6))
    sns.histplot(df['times_viewed'], bins=50, kde=True)
    plt.title('Distribution of Times Viewed')
    plt.xlabel('Times Viewed')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_top_countries_revenue(df):
    """Plot top 10 countries by total revenue."""
    top_countries = df.groupby('country')['price'].sum().nlargest(10).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='price', y='country', data=top_countries, palette='viridis')
    plt.title('Top 10 Countries by Total Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('Country')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    df = load_data(DATA_DIR)
    plot_time_series(df)
    plot_time_series(df, country='United Kingdom')
    plot_revenue_distribution(df)
    plot_views_distribution(df)
    plot_top_countries_revenue(df)
