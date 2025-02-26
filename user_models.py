from werkzeug.security import generate_password_hash, check_password_hash
from database import mongo

class User:
    @staticmethod
    def create_user(username, password, role='user'):
        hashed_password = generate_password_hash(password)
        user = {'username': username, 'password': hashed_password, 'role': role}
        mongo.db.users.insert_one(user)
        return user

    @staticmethod
    def authenticate(username, password):
        user = mongo.db.users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            return user
        return None
