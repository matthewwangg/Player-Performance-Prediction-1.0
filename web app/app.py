from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

app = Flask(__name__, static_url_path="/static")

@app.route('/')
def index():

    predicts()
    return render_template('index.html')

#@app.route('/predicts', methods=['GET','POST'])
def predicts():
    
    # Reading in the CSV file from Kaggle (Credits to Paola Mazza) into a Pandas Data Frame
    players_df = pd.read_csv("players.csv")

    positions = ["DEF", "MID", "FWD", "GKP"]

    # Preprocess and separate the dataframe
    dataframes = preprocess(players_df, positions)

    # Make predictions using your function from the module
    models = train_models(dataframes, positions)

    for i in range(len(models)):
        evaluate_model(models[i], dataframes[i].select_dtypes(include=['int']).drop(columns=['total_points']), dataframes[i]['total_points'], positions[i])


# Preprocess the data within the Pandas Data Frame
def preprocess(players_df, positions):

    split_dataframes = []

    # Split the dataset based on values in the 'Position' column
    for i in positions:
        df = players_df[players_df['position'] == i]
        split_dataframes.append(df)

    for i in range(len(split_dataframes)):
    	split_dataframes[i] = split_dataframes[i].copy()
    	split_dataframes[i] = split_dataframes[i].drop_duplicates()
  
    return split_dataframes

# Function to train all 4 models for each position
def train_models(dataframes, positions):

    trained_models = []

    for i in range(len(dataframes)):
        integer_columns = dataframes[i].select_dtypes(include=['int']).drop(columns=['total_points'])
        xgb_model = train_xgboost_model(integer_columns, dataframes[i]['total_points'], positions[i])
        trained_models.append(xgb_model)

    return trained_models

# Function to train XGBoost model with the predefined hyperparameters
def train_xgboost_model(X, y, position):

    # Define hyperparameters
    hyperparams = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100
    }

    # Create XGBoost regressor with predefined hyperparameters
    xgb_model = XGBRegressor(**hyperparams)

    # Train the model
    xgb_model.fit(X, y)

    print(f"XGBoost Model trained for {position} position.")

    return xgb_model

# Function to evaluate XGBoost model and print MSE
def evaluate_model(model, X_test, y_test, position):

    # Get the predictions
    y_pred = model.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error for {position}: {mse}")

# Function to perform hyperparameter tuning with cross-validation
def tune_hyperparameters(X, y):

    # Define the parameter grid
    param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'n_estimators': [100, 200]
    }

    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor()

    # Perform grid search with cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
    grid_search.fit(X, y)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    return best_params

# Function to train XGBoost model with the best hyperparameters
def train_xgboost_model_best(X, y, position):

    # Perform hyperparameter tuning
    best_params = tune_hyperparameters(X, y)

    # Create XGBoost regressor with best hyperparameters
    xgb_model = XGBRegressor(**best_params)

    # Train the model
    xgb_model.fit(X, y)

    print(f"XGBoost Model trained for {position} position.")

    return xgb_model

if __name__ == '__main__':
    app.run(debug=True)
