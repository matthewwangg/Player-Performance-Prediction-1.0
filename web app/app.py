from flask import Flask, request, render_template
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
import requests
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
from xgboost import plot_importance
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpMinimize
from data_processing import predicts

app = Flask(__name__, static_url_path="/static")

@app.route('/')
def index():

    return render_template('index.html')
    #top_players, optimized_players = predicts()
    #return render_template('index.html', top_players=top_players, optimized_players=optimized_players)

@app.route('/predict', methods=['POST'])
def predict():
    # Call your predict function here and store the results
    top_players, optimized_players = predicts()

    return render_template('index.html', top_players=top_players, optimized_players=optimized_players)

    # SETUP for future Javascript Functionality: Return a response that can be handled by the frontend
    # return jsonify({'top_players': top_players, 'optimized_players': optimized_players})

if __name__ == '__main__':
    app.run(debug=True)
