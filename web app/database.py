from flask_sqlalchemy import SQLAlchemy
import os
import pandas as pd  # Make sure Pandas is imported

db = SQLAlchemy()

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    team = db.Column(db.String(40), nullable=True)
    position = db.Column(db.String(4), nullable=True)

def populate_database(csv_file_path):
    # Load the CSV file
    players_df = pd.read_csv(csv_file_path)

    # Iterate through the DataFrame and add each player to the database
    for index, row in players_df.iterrows():
        player = Player(
            name=row['name'],
            team=row['team'] if 'team' in row and pd.notnull(row['team']) else None,
            position=row['position'] if 'position' in row and pd.notnull(row['position']) else None
        )
        db.session.add(player)

    # Commit the changes
    db.session.commit()

def init_app(app):
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL').replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()
