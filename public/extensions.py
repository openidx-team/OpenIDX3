from flask import Flask
from flask_sqlalchemy import SQLAlchemy

import os
import random
import threading
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = random._urandom(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'stocks.db')
db = SQLAlchemy(app)