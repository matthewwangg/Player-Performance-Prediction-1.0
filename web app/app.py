from flask import Flask, request, render_template
from data_processing import predicts, predicts_custom
from data_pipeline import initialize_map


app = Flask(__name__, static_url_path="/static")

# Initialize the database with the Flask app
#database.init_app(app)

@app.route('/')
def index():

    initialize_map()
    return render_template('index.html')
    #top_players, optimized_players = predicts()
    #return render_template('index.html', top_players=top_players, optimized_players=optimized_players)

@app.route('/predict', methods=['POST'])
def predict():

    top_players, optimized_players = predicts()

    return render_template('index.html', top_players=top_players, optimized_players=optimized_players)

@app.route('/predict_custom', methods=['POST'])
def predict_custom():

    data = request.json

    optimized_players = predicts_custom(data)

    return render_template('index.html', optimized_players_custom=optimized_players)

if __name__ == '__main__':
    app.run(debug=True)
