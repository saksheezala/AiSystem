from flask_pymongo import PyMongo
from config import Config

mongo = PyMongo()

def init_db(app):
    app.config.from_object(Config)
    mongo.init_app(app)
    return mongo
