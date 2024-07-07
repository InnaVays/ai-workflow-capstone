import warnings
warnings.filterwarnings("ignore", message="resource_tracker: .*")

import time,os,re,csv,sys,uuid,joblib
from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

import sys
sys.path.append("model")
from logger import update_predict_log, update_train_log
from cslib import fetch_ts, engineer_features

## model specific variables (iterate the version and note with each change)
MODEL_DIR = "models"
MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "supervised learning model for time-series"

def _model_train(df,tag,test=False):

    ## start timer for runtime
    time_start = time.time()
    
    X,y,dates = engineer_features(df)

    if test:
        n_samples = int(np.round(0.3 * X.shape[0]))
        subset_indices = np.random.choice(np.arange(X.shape[0]),n_samples,
                                          replace=False).astype(int)
        mask = np.in1d(np.arange(y.size),subset_indices)
        y=y[mask]
        X=X[mask]
        dates=dates[mask]
        
    ## Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        shuffle=True, random_state=42)
    ## train a random forest model
    param_grid_rf = {
    'rf__criterion': ['squared_error', 'absolute_error'],
    'rf__n_estimators': [10, 15, 20, 25]
}


    pipe_rf = Pipeline(steps=[('scaler', StandardScaler()),
                              ('rf', RandomForestRegressor())])
    
    grid = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    eval_rmse =  round(np.sqrt(mean_squared_error(y_test,y_pred)))
    
    ## retrain using all data
    grid.fit(X, y)
    model_name = re.sub("\.","_",str(MODEL_VERSION))
    if test:
        saved_model = os.path.join(MODEL_DIR,
                                   "test-{}-{}.joblib".format(tag,model_name))
        print("... saving test version of model: {}".format(saved_model))
    else:
        saved_model = os.path.join(MODEL_DIR,
                                   "sl-{}-{}.joblib".format(tag,model_name))
        print("... saving model: {}".format(saved_model))
        
    joblib.dump(grid,saved_model)

    m, s = divmod(time.time()-time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d"%(h, m, s)

    ## update log
    update_train_log(tag,(str(dates[0]),str(dates[-1])),{'rmse':eval_rmse},runtime,
                     MODEL_VERSION, MODEL_VERSION_NOTE,test=True)
  
def model_train(data_dir,test=False):
    """
    function to train model given a df
    
    'mode' -  can be used to subset data essentially simulating a train
    """
    
    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if test:
        print("... test flag on")
        print("...... subsetting data")
        print("...... subsetting countries")
        
    ## fetch time-series formatted data
    ts_data = fetch_ts(data_dir)

    ## train a different model for each data sets
    for country,df in ts_data.items():
        
        if test and country not in ['all','united_kingdom']:
            continue
        
        _model_train(df,country,test=test)
    
def model_load(prefix='sl',data_dir=None,training=True):
    """
    example funtion to load model
    
    The prefix allows the loading of different models
    """

    if not data_dir:
        data_dir = os.path.join("data","cs-train")
    
    models = [f for f in os.listdir(os.path.join("models")) if re.search(prefix,f)]

    if len(models) == 0:
        raise Exception("Models with prefix '{}' cannot be found did you train?".format(prefix))

    all_models = {}
    for model in models:
        all_models[re.split("-",model)[1]] = joblib.load(os.path.join("models",model))

    ## load data
    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country, df in ts_data.items():
        df = clean_data(df)
        X,y,dates = engineer_features(df,training=training)
        dates = np.array([str(d) for d in dates])
        all_data[country] = {"X":X,"y":y,"dates": dates}
        
    return(all_data, all_models)
    
def clean_data(df):
    revenue = df['revenue']
    
    if df.empty:
        return(df)
    
    # without negative revenue
    mask = revenue < 0
    df.loc[mask, 'revenue'] = np.nan  # Changed from df.at to df.loc

    # without quantile range outliers
    mask = revenue.between(revenue.quantile(.0), revenue.quantile(0.85)) 
    df.loc[~mask, 'revenue'] = np.nan  # Changed from df.at to df.loc

    mask = revenue.notna()
    median = np.median(df[mask]['revenue'])

    #df['revenue'].fillna(median, inplace = True)
    df.fillna({'revenue': median}, inplace=True)
    
    return(df)

def model_predict(country, year, month, day, all_models=None, test=False):
    """
    example function to predict from model
    """

    ## start timer for runtime
    time_start = time.time()

    ## load model if needed
    if not all_models:
        data_dir = os.path.join("data", "cs-train") if test else None
        all_data, all_models = model_load(prefix="test" if test else "sl", data_dir=data_dir, training=False)
    else:
        # If all_models is provided, we need to load all_data separately
        data_dir = os.path.join("data", "cs-train") if test else None
        all_data, _ = model_load(prefix="test" if test else "sl", data_dir=data_dir, training=False)
    
    ## input checks
    country_key = re.sub(r"\s+", "_", country.lower())
    if country_key not in all_models.keys():
        raise Exception(f"ERROR (model_predict) - model for country '{country}' could not be found")

    for d in [year, month, day]:
        if re.search("\D", d):
            raise Exception(f"ERROR (model_predict) - invalid year, month, or day: {d}")
    
    ## load data
    model = all_models[country_key]
    data = all_data[country_key]

    ## check date
    target_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    if target_date not in data['dates']:
        raise Exception(f"ERROR (model_predict) - date {target_date} not in range {data['dates'][0]}-{data['dates'][-1]}")
    date_indx = np.where(data['dates'] == target_date)[0][0]
    query = data['X'].iloc[[date_indx]]
    
    ## sanity check
    if data['dates'].shape[0] != data['X'].shape[0]:
        raise Exception("ERROR (model_predict) - dimensions mismatch")

    ## make prediction and gather data for log entry
    y_pred = model.predict(query)
    y_proba = None
    if 'predict_proba' in dir(model) and 'probability' in dir(model):
        if model.probability:
            y_proba = model.predict_proba(query)

    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    runtime = "%03d:%02d:%02d" % (h, m, s)

    ## update predict log
    update_predict_log(country, y_pred, y_proba, target_date, runtime, MODEL_VERSION, test=test)
    
    return {'y_pred': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
            'y_proba': y_proba.tolist() if isinstance(y_proba, np.ndarray) else y_proba}

if __name__ == "__main__":
    """
    basic test procedure for model.py
    """

    ## train the model
    print("TRAINING MODELS")
    data_dir = os.path.join("data", "cs-train")
    model_train(data_dir, test=True)

    ## load the model
    print("LOADING MODELS")
    all_data, all_models = model_load(prefix="test")
    print("... models loaded: ", ",".join(all_models.keys()))

    ## test predict
    country = 'all'
    year = '2018'
    month = '01'
    day = '05'
    result = model_predict(country, year, month, day, all_models=all_models, test=True)
    print(result)