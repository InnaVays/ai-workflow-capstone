# IBM AI Enterprise Workflow Capstone
That is a fork of the IBM AI Enterprise Workflow Capstone project. 

# FastAPI ML Model API

This repository contains a FastAPI-based API for training and predicting machine learning models for time-series data. The project includes Docker support, unit tests, and post-production analysis scripts.

## Getting Started

### Prerequisites

- Docker
- Docker Compose
- Python 3.10+

### Installation
1. Build and run the Docker container:
   docker-compose up --build
2. Ensure the FastAPI server is running:
   docker ps
3. Run
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload.
4. Docker Support
   The project includes Docker support for easy deployment. The docker_compose.yml file sets up the necessary services.
5. Stop the Docker Container:
   docker-compose down

## API Endpoints
### Train the Model
To train the model, send a POST request to the /train endpoint with the appropriate data directory.

### Make a Prediction
To make a prediction, send a POST request to the /predict endpoint with the required parameters.

## Running Unit Tests
To run the unit tests, use the following command:

```sh
pytest run_tests.py
```

## Batch Prediction
For batch prediction, use the batch_prediction.py script. Ensure the FastAPI server is running and then execute the script:

```sh
python batch_prediction.py
```

## Exploratory Data Analysis (EDA)
To perform exploratory data analysis, use the eda.py script. This script helps visualize the data and analyze various aspects of the dataset.

```sh
pytest eda.py
```

## Post-Production Analysis
The analysis.py script investigates the relationship between model performance and business metrics. 

```sh
pytest analysis.py
```

