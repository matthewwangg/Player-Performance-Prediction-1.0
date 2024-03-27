from flask import Flask, request, render_template
from data_processing import predicts
from data_pipeline import initialize_map


app = Flask(__name__, static_url_path="/static")

# Initialize the database with the Flask app
# Commented for now
#database.init_app(app)

@app.route('/')
def index():

    initialize_map()
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
