import os
import pandas as pd
from model_training import train_models, evaluate_model, visualize
from optimizations import linear_optimization


# @app.route('/predicts', methods=['GET','POST'])
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
        evaluate_model(models[i], dataframes[i].select_dtypes(include=['int']).drop(columns=['total_points']),
                       dataframes[i]['total_points'], positions[i])

    visualize(models, "visualizations", positions)

    top_players = []

    for i in range(len(models)):
        top_players.extend(get_top_players(models[i], dataframes[i], positions[i], count[i]))

    predicted_points_df = create_predicted_points_and_costs_dataframe(models, positions, players_df)

    optimized_players = linear_optimization(predicted_points_df)

    # Convert optimized players to list of dicts for easier template rendering
    optimized_players_list = optimized_players.to_dict(orient='records')

    print(optimized_players_list)

    return top_players, optimized_players_list


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

    # Get the player names and positions
    player_names = dataframe['name']
    positions = dataframe['position']

    # Create a DataFrame with player names, positions, and their predicted points
    player_points_df = pd.DataFrame({'name': player_names, 'position': positions, 'predicted_points': predictions})

    return player_points_df

# Function to create a DataFrame with predicted points and costs for all players
def create_predicted_points_and_costs_dataframe(models, positions, players_df):
    # Create an empty list to store DataFrames for each position
    dfs = []

    # Iterate through each position and corresponding model
    for i in range(len(models)):
        # Filter players_df for current position
        position_df = players_df[players_df['position'] == positions[i]]

        # Get predicted points for players in this position
        predicted_points_df = get_predicted_points_for_position(models[i], position_df)

        # Merge costs into predicted_points_df based on player names (or IDs if available)
        predicted_points_df = predicted_points_df.merge(position_df[['name', 'now_cost']], on='name', how='left')

        # Append the DataFrame to the list
        dfs.append(predicted_points_df)

    # Concatenate DataFrames for each position
    predicted_points_and_costs_df = pd.concat(dfs, ignore_index=True)

    return predicted_points_and_costs_df

