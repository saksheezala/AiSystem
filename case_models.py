from database import mongo
from bson import ObjectId

class Case:
    @staticmethod
    def create_case(user_id, title, description):
        case = {
            'user_id': user_id,
            'title': title,
            'description': description,
            'images': []
        }
        result = mongo.db.cases.insert_one(case)
        return str(result.inserted_id)

    @staticmethod
    def get_cases(user_id):
        return list(mongo.db.cases.find({'user_id': user_id}))

    @staticmethod
    def get_case(case_id):
        return mongo.db.cases.find_one({'_id': ObjectId(case_id)})

    @staticmethod
    def add_image(case_id, image_data):
        mongo.db.cases.update_one({'_id': ObjectId(case_id)}, {'$push': {'images': image_data}})
