from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGO_URI"))
db = client.get_database("your_database_name")  # Change to your database name
users_collection = db["users"]

# Admin credentials
admin_username = "admin"
admin_password = "admin123"  # Change this to a strong password

# Check if admin already exists
if users_collection.find_one({"username": admin_username}):
    print("Admin user already exists.")
else:
    # Hash the password
    hashed_password = generate_password_hash(admin_password)

    # Insert admin user
    users_collection.insert_one({
        "username": admin_username,
        "password": hashed_password,
        "role": "admin"
    })
    print("Admin user created successfully.")
