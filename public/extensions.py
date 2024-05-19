"""
This module contains the Flask application and the SQLAlchemy database object.
"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = os.urandom(16)
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'sqlite:///' + os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'database', 'stocks.db'
    )
)
db = SQLAlchemy(app)
