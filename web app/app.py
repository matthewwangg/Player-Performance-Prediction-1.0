from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
import requests
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
from xgboost import plot_importance


app = Flask(__name__, static_url_path="/static")

@app.route('/')
def index():

    top_players = predicts()
    return render_template('index.html', top_players=top_players)

#@app.route('/predicts', methods=['GET','POST'])
def predicts():

    # Reading in the CSV file from Kaggle (Credits to Paola Mazza) into a Pandas Data Frame
    players_df = pd.read_csv(find_path())

    positions = ["DEF", "MID", "FWD", "GKP"]
    count = [5, 5, 3, 2]

    # Preprocess and separate the dataframe
    dataframes = preprocess(players_df, positions)

    # Make predictions using your function from the module
    models = train_models(dataframes, positions)

    for i in range(len(models)):
        evaluate_model(models[i], dataframes[i].select_dtypes(include=['int']).drop(columns=['total_points']), dataframes[i]['total_points'], positions[i])

    visualize(models, "visualizations", positions)

    top_players = []

    for i in range(len(models)):
        top_players.extend(get_top_players(models[i], dataframes[i], positions[i], count[i]))

    for i in range(len(models)):
        print(get_predicted_points_for_position(models[i], dataframes[i]).head())

    return top_players

def find_path():
    # Get the current directory of the Flask application
    current_dir = os.path.dirname(__file__)

    # Navigate to the parent directory
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    # Define the path to the CSV file
    csv_file_path = os.path.join(parent_dir, 'datasets', 'players.csv')
    return csv_file_path

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

# Get the top players
def get_top_players(model, dataframe, position, n):
    # Get the predictions
    X = dataframe.select_dtypes(include=['int']).drop(columns=['total_points'])
    predictions = model.predict(X)

    # Get the player names
    player_names = dataframe['name']

    # Create a DataFrame with player names and their predicted points
    player_points_df = pd.DataFrame({'name': player_names, 'predicted_points': predictions})

    # Sort the DataFrame by predicted points in descending order
    sorted_df = player_points_df.sort_values(by='predicted_points', ascending=False)

    # Get the top n players
    top_players = sorted_df.head(n)

    return top_players.values.tolist()

# Function to get predicted points for players in each position
def get_predicted_points_for_position(model, dataframe):
    # Assuming 'dataframe' is the input dataframe containing player information
    X = dataframe.select_dtypes(include=['int']).drop(columns=['total_points'])
    predictions = model.predict(X)

    # Get the player names
    player_names = dataframe['name']

    # Create a DataFrame with player names and their predicted points
    player_points_df = pd.DataFrame({'name': player_names, 'predicted_points': predictions})

    return player_points_df

# Function to create a DataFrame with predicted points and costs for all players
def create_predicted_points_and_costs_dataframe(models, positions):
    # Create an empty list to store DataFrames for each position
    dfs = []

    # Iterate through each position and corresponding model
    for i in range(len(models)):
        # Get predicted points for players in this position
        # Here, you need the input dataframe containing player information
        predicted_points_df = get_predicted_points_for_position(models[i], players_df[players_df['position'] == positions[i]])

        # Append the DataFrame to the list
        dfs.append(predicted_points_df)

    # Concatenate DataFrames for each position
    predicted_points_and_costs_df = pd.concat(dfs, ignore_index=True)

    # Assuming 'players_df' is the input dataframe containing player information
    # Merge player costs from the original dataframe
    predicted_points_and_costs_df['cost'] = players_df['now_cost']

    return predicted_points_and_costs_df

# Function to get the manifest for the player images
def get_manifest_json():
    sport = "soccer"
    access_level = "t"
    version = "3"
    provider = "reuters"
    league = "epl"
    image_type = "headshots"
    year = "2024"
    format = "json"
    your_api_key = "muy5mttxmvb4zx6q995w4zcf"
    manifest_search_url = f"https://api.sportradar.us/{sport}-images-{access_level}{version}/{provider}/{league}/{image_type}/players/{year}/manifest.{format}?api_key={your_api_key}"

    try:
        # Make a request to the image search API
        response = requests.get(manifest_search_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        image_results = response.json()

        # Extract image URLs from the API response
        image_urls = [result["url"] for result in image_results["results"]]
        print(image_urls)
        return image_urls

    except requests.RequestException as e:
        print(f"Error fetching image URLs: {e}")
        return []

# Function to get the image urls given the player names
def get_image_urls(top_players):
    image_urls = []
    sport = "soccer"
    access_level = ""
    version = "3"
    provider = "reuters"
    league = "epl"
    image_type = "headshots"
    year = "2024"
    format = "json"
    your_api_key = "3wtfjw2jfy2wwuwbudr6b9ru"
    imagesearchurl = "https://api.sportradar.us/{sport}-images-{access_level}{version}/{provider}/{league}/{image_type}/players/{asset_id}/{file_name}.{format}?api_key={your_api_key}"

# Function to train XGBoost model with the predefined hyperparameters
def train_xgboost_model(X, y, position):

    # Define hyperparameters
    hyperparams = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_estimators': 100,
        'alpha': 0.1,
        'lambda': 0.1
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


def visualize(models, output_dir, positions):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    visualizations = []

    for idx, model in enumerate(models):
        # Generate the feature importance plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_importance(model, ax=ax)

        # Save the plot as an image file
        image_path = os.path.join(output_dir, f"visualization_{positions[idx]}.png")
        fig.savefig(image_path, format='png')

        # Close the figure to release memory
        plt.close(fig)

        # Append the image path to the list of visualizations
        visualizations.append(image_path)

    return visualizations

# Function to set the parameters for the linear optimization
def linear_optimization():

    # Set the maximum budget and position constraints
    max_budget = 1000
    max_keepers = 2
    max_defenders = 5
    max_midfielders = 5
    max_forwards = 3

    # Call the optimize_team function
    selected_team = optimize_team(players_df, max_budget, max_keepers, max_defenders, max_midfielders, max_forwards)

    # Display the selected team
    print("Selected Team:")
    print(selected_team)

# Function that uses PuLP to optimize team
def optimize_team(players_df, max_cost, max_keepers, max_defenders, max_midfielders, max_forwards):
    # Create a linear programming problem
    prob = LpProblem("TeamOptimization", LpMaximize)

    # Create binary decision variables for each player
    players_df['selected'] = LpVariable.dicts("Player", players_df.index, cat="Binary")

    # Objective function: Maximize total points
    prob += lpSum(players_df['total_points'] * players_df['selected'])

    # Cost constraint: Total cost should be less than max_cost
    prob += lpSum(players_df['now_cost'] * players_df['selected']) <= max_cost

    # Position constraints: Limit the number of players from each role
    prob += lpSum(players_df['selected'][players_df['position'] == 'GKP']) <= max_keepers
    prob += lpSum(players_df['selected'][players_df['position'] == 'DEF']) <= max_defenders
    prob += lpSum(players_df['selected'][players_df['position'] == 'MID']) <= max_midfielders
    prob += lpSum(players_df['selected'][players_df['position'] == 'FWD']) <= max_forwards

    # Solve the problem
    prob.solve()

    # Extract the selected players
    selected_players = players_df.loc[players_df['selected'].apply(lambda x: x.varValue) == 1]

    return selected_players


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
