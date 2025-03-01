from flask import Flask, render_template, redirect, url_for, request, session, flash, send_from_directory, Response, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from dotenv import load_dotenv
import os
import gridfs
import io
import datetime
from database import init_db, mongo
from user_models import User
from case_models import Case
from model import classify_image
from pymongo import MongoClient

# New imports for enhanced security & notifications
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.urandom(24)

# Initialize MongoDB connection
init_db(app)

# Initialize GridFS
fs = gridfs.GridFS(mongo.db)

# Set up Flask-Limiter for rate limiting (e.g., 100 requests per hour per IP)
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per hour"])
limiter.init_app(app)

# Set up Flask-Mail configuration (update with your .env values)
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', '1', 'yes']
app.config['MAIL_USE_SSL'] = os.getenv('MAIL_USE_SSL', 'false').lower() in ['true', '1', 'yes']
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'saksheezala1111@gmail.com')
mail = Mail(app)

# --------------------
# Error Handlers
# --------------------
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

# --------------------
# Activity Logs Helper
# --------------------
def log_activity(action, details=""):
    log_entry = {
        'user_id': session.get('user_id', None),
        'username': session.get('username', None),
        'action': action,
        'details': details,
        'timestamp': datetime.datetime.utcnow()
    }
    mongo.db.activity_logs.insert_one(log_entry)

# --------------------
# Existing Routes
# --------------------
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = mongo.db.users.find_one({'username': username})
        if user and check_password_hash(user['password'], password):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['role'] = user['role']
            log_activity("Login", f"User {username} logged in.")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({
            'username': username,
            'password': hashed_password,
            'role': 'user'
        })
        log_activity("Registration", f"User {username} registered.")
        flash('Registration successful. Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session['role'] == 'admin':
        users = list(mongo.db.users.find())
        for user in users:
            user['_id'] = str(user['_id'])
        return render_template('user_list.html', users=users)
    else:
        page = request.args.get('page', 1, type=int)
        per_page = 5
        skip_count = (page - 1) * per_page
        query = {'user_id': session['user_id'], 'deleted': {"$ne": True}}
        total = mongo.db.cases.count_documents(query)
        cases_cursor = mongo.db.cases.find(query).skip(skip_count).limit(per_page)
        cases = list(cases_cursor)
        total_pages = (total + per_page - 1) // per_page
        return render_template('dashboard.html', cases=cases, page=page, total_pages=total_pages)

@app.route('/add_case', methods=['GET', 'POST'])
def add_case():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        case_id = mongo.db.cases.insert_one({
            'title': title,
            'description': description,
            'user_id': session['user_id'],
            'images': []
        }).inserted_id
        log_activity("Case Created", f"Case '{title}' (ID: {case_id}) created by {session['username']}.")
        flash('Case added successfully.')
        return redirect(url_for('dashboard'))
    return render_template('add_case.html')

@app.route('/upload_image/<case_id>', methods=['GET', 'POST'])
def upload_img(case_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        image = request.files['image']
        if image:
            upload_folder = os.path.join(os.getcwd(), "uploads")
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)
            image_path = os.path.join(upload_folder, image.filename)
            image.save(image_path)
            print("Image saved at:", image_path)  # Debug print

            try:
                with open(image_path, 'rb') as f:
                    image_result = classify_image(f)
            except Exception as e:
                image_result = f"Error: Invalid image file. {str(e)}"
            print("Classification result:", image_result)  # Debug print

            mongo.db.cases.update_one(
                {'_id': ObjectId(case_id)},
                {'$push': {'images': {'path': os.path.join("uploads", image.filename), 'type': image_result}}}
            )
            log_activity("Image Uploaded", f"Image '{image.filename}' uploaded to case {case_id}, classified as {image_result}.")

            if image_result != "Attack Type: Normal":
                now = datetime.datetime.utcnow()
                last_alert = session.get('last_alert_time')
                if not last_alert or (now - last_alert).total_seconds() > 1800:
                    try:
                        msg = Message("Adversarial Attack Detected", recipients=[os.getenv("ADMIN_EMAIL")])
                        msg.body = f"An adversarial attack was detected in case {case_id}.\nClassification: {image_result}."
                        mail.send(msg)
                        session['last_alert_time'] = now
                        log_activity("Email Notification", f"Sent alert email for case {case_id} with classification {image_result}.")
                    except Exception as e:
                        log_activity("Email Error", f"Error sending email for case {case_id}: {str(e)}")
            flash(f'Image classified as {image_result}.')
            return redirect(url_for('view_case', case_id=case_id))
    return render_template('upload_image.html', case_id=case_id)



@app.route('/delete_my_case/<case_id>', methods=['POST'])
def delete_my_case(case_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    mongo.db.cases.update_one(
        {'_id': ObjectId(case_id), 'user_id': session['user_id']},
        {'$set': {'deleted': True}}
    )
    log_activity("Case Deleted", f"User {session['username']} soft-deleted case {case_id}.")
    flash('Case deleted (soft delete).')
    return redirect(url_for('dashboard'))

@app.route('/delete_my_image_from_path/<case_id>/<filename>', methods=['POST'])
def delete_my_image_from_path(case_id, filename):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    # Construct the expected path using os.path.join for consistency
    expected_path = os.path.join("uploads", filename)
    
    # Use array filters to update all matching array elements
    result = mongo.db.cases.update_one(
        {'_id': ObjectId(case_id), 'user_id': session['user_id']},
        {'$set': {'images.$[elem].deleted': True}},
        array_filters=[{'elem.path': expected_path}]
    )
    
    if result.modified_count:
        log_activity("Image Deleted", f"User {session['username']} soft-deleted image '{filename}' from case {case_id}.")
        flash('Image deleted (soft delete).')
    else:
        flash("Image deletion did not modify any document. Please check the image details.")
    
    return redirect(url_for('view_case', case_id=case_id))


@app.route('/get_uploaded_image/<image_id>')
def get_uploaded_image(image_id):
    try:
        image_file = fs.get(ObjectId(image_id))
        return Response(image_file.read(), mimetype='image/jpeg')
    except Exception as e:
        return f"Error retrieving image: {str(e)}", 404

@app.route('/uploads/<filename>')
def get_image_from_path(filename):
    uploads_path = os.path.join(os.getcwd(), "uploads")
    file_path = os.path.join(uploads_path, filename)
    print("Serving image from:", file_path)  # Debug: Check this in your logs
    return send_from_directory(uploads_path, filename)


@app.route('/view_case/<case_id>')
def view_case(case_id):
    case = mongo.db.cases.find_one({'_id': ObjectId(case_id)})
    if case and session.get('role') != 'admin':
        case['images'] = [img for img in case.get('images', []) if not img.get('deleted')]
    return render_template('case_detail.html', case=case)

@app.route('/user_cases/<user_id>')
def user_cases(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    user_doc = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    username = user_doc.get('username') if user_doc else user_id
    cases = list(mongo.db.cases.find({'user_id': user_id}))
    return render_template('user_cases.html', cases=cases, username=username)

@app.route('/delete_user/<user_id>', methods=['POST'])
def delete_user(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    mongo.db.users.delete_one({'_id': ObjectId(user_id)})
    mongo.db.cases.delete_many({'user_id': user_id})
    log_activity("User Deleted", f"Admin {session['username']} deleted user {user_id}.")
    flash('User deleted successfully.')
    return redirect(url_for('dashboard'))

@app.route('/delete_case/<case_id>', methods=['POST'])
def delete_case(case_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    mongo.db.cases.delete_one({'_id': ObjectId(case_id)})
    log_activity("Case Permanently Deleted", f"Admin {session['username']} permanently deleted case {case_id}.")
    flash('Case permanently deleted.')
    return redirect(url_for('dashboard'))

@app.route('/admin_delete_image/<case_id>/<image_id>', methods=['POST'])
def admin_delete_image(case_id, image_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    mongo.db.cases.update_one(
        {'_id': ObjectId(case_id)},
        {'$pull': {'images': {'image_id': image_id}}}
    )
    log_activity("Image Permanently Deleted", f"Admin {session['username']} permanently deleted image {image_id} from case {case_id}.")
    flash('Image permanently deleted.')
    return redirect(url_for('view_case', case_id=case_id))

@app.route('/profile', methods=['GET'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    if user:
        user['_id'] = str(user['_id'])
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    new_username = request.form.get('username')
    new_email = request.form.get('email')
    new_password = request.form.get('password')
    update_fields = {}
    if new_username:
        update_fields['username'] = new_username
    if new_email:
        update_fields['email'] = new_email
    if new_password:
        update_fields['password'] = generate_password_hash(new_password)
    mongo.db.users.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': update_fields}
    )
    if new_username:
        session['username'] = new_username
    log_activity("Profile Updated", f"User {session['username']} updated their profile.")
    flash('Profile updated successfully.')
    return redirect(url_for('profile'))

@app.route('/edit_case/<case_id>', methods=['GET', 'POST'])
def edit_case(case_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    case = mongo.db.cases.find_one({'_id': ObjectId(case_id), 'user_id': session['user_id']})
    if not case:
        flash("Case not found or access denied.")
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        new_title = request.form.get('title')
        new_description = request.form.get('description')
        mongo.db.cases.update_one(
            {'_id': ObjectId(case_id)},
            {'$set': {'title': new_title, 'description': new_description}}
        )
        log_activity("Case Updated", f"User {session['username']} updated case {case_id}.")
        flash("Case updated successfully.")
        return redirect(url_for('view_case', case_id=case_id))
    return render_template('edit_case.html', case=case)

@app.route('/edit_image/<case_id>/<identifier>', methods=['GET', 'POST'])
def edit_image(case_id, identifier):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    case = mongo.db.cases.find_one({'_id': ObjectId(case_id), 'user_id': session['user_id']})
    if not case:
        flash("Case not found or access denied.")
        return redirect(url_for('dashboard'))
    image_data = None
    image_index = None
    for idx, img in enumerate(case.get('images', [])):
        if (img.get('image_id') and img['image_id'] == identifier) or \
           (img.get('path') and img['path'].split('/')[-1] == identifier):
            image_data = img
            image_index = idx
            break
    if image_data is None:
        flash("Image not found.")
        return redirect(url_for('view_case', case_id=case_id))
    if request.method == 'POST':
        new_comment = request.form.get('comment')
        result = mongo.db.cases.update_one(
            {'_id': ObjectId(case_id)},
            {'$set': {f'images.{image_index}.comment': new_comment}}
        )
        if result.modified_count == 0:
            flash("No changes were made. Please try again or check the data.")
        else:
            flash("Image comment updated.")
            log_activity("Image Comment Updated", f"User {session['username']} updated comment for image in case {case_id}.")
        return redirect(url_for('view_case', case_id=case_id))
    return render_template('edit_image.html', case=case, image=image_data, identifier=identifier)

@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    pipeline = [
        {'$match': {'user_id': session['user_id'], 'deleted': {"$ne": True}}},
        {'$unwind': '$images'},
        {'$match': {'images.deleted': {"$ne": True}}},
        {'$group': {'_id': '$images.type', 'count': {'$sum': 1}}}
    ]
    results = list(mongo.db.cases.aggregate(pipeline))
    data = {result['_id']: result['count'] for result in results}
    classifications = ['Attack Type: Normal', 'Attack Type: FGSM', 'Attack Type: PGD', 'Unknown']
    counts = [data.get(cls, 0) for cls in classifications]
    return render_template('analytics.html', classifications=classifications, counts=counts)

@app.route('/analytics/<case_id>')
def analytics_per_case(case_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    case = mongo.db.cases.find_one({'_id': ObjectId(case_id)})
    if not case:
        flash("Case not found.")
        return redirect(url_for('dashboard'))
    case_title = case.get('title', case_id)
    pipeline = [
        {'$match': {'_id': ObjectId(case_id)}},
        {'$unwind': '$images'},
        {'$match': {'images.deleted': {"$ne": True}}},
        {'$group': {'_id': '$images.type', 'count': {'$sum': 1}}}
    ]
    results = list(mongo.db.cases.aggregate(pipeline))
    data = {result['_id']: result['count'] for result in results}
    classifications = ['Attack Type: Normal', 'Attack Type: FGSM', 'Attack Type: PGD', 'Unknown']
    counts = [data.get(cls, 0) for cls in classifications]
    return render_template('analytics.html', classifications=classifications, counts=counts, case_id=case_id, case_title=case_title)

@app.route('/activity_logs')
def activity_logs():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('login'))
    logs = list(mongo.db.activity_logs.find().sort('timestamp', -1))
    for log in logs:
        log['_id'] = str(log['_id'])
        log['timestamp'] = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
    return render_template('activity_logs.html', logs=logs)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --------------------
# API Endpoints
# --------------------
@app.route('/api/cases', methods=['GET'])
def api_get_cases():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    user_id = session['user_id']
    if session.get('role') == 'admin':
        cases_cursor = mongo.db.cases.find()
    else:
        cases_cursor = mongo.db.cases.find({'user_id': user_id, 'deleted': {"$ne": True}})
    cases = []
    for case in cases_cursor:
        case['_id'] = str(case['_id'])
        if session.get('role') != 'admin':
            case['images'] = [img for img in case.get('images', []) if not img.get('deleted')]
        cases.append(case)
    return jsonify(cases), 200

@app.route('/api/cases/<case_id>', methods=['GET'])
def api_get_case(case_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    case = mongo.db.cases.find_one({'_id': ObjectId(case_id)})
    if not case:
        return jsonify({"error": "Case not found"}), 404
    case['_id'] = str(case['_id'])
    if session.get('role') != 'admin':
        case['images'] = [img for img in case.get('images', []) if not img.get('deleted')]
    return jsonify(case), 200

@app.route('/api/upload_image/<case_id>', methods=['POST'])
def api_upload_image(case_id):
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image = request.files['image']
    if image:
        image_result = classify_image(image)
        mongo.db.cases.update_one(
            {'_id': ObjectId(case_id)},
            {'$push': {'images': {'path': os.path.join("uploads", image.filename), 'type': image_result}}}
        )
        log_activity("API Image Uploaded", f"Image '{image.filename}' uploaded to case {case_id} via API, classified as {image_result}.")
        return jsonify({"result": image_result}), 200
    return jsonify({"error": "Image upload failed"}), 500

@app.route('/api/update_profile', methods=['POST'])
def api_update_profile():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    update_fields = {}
    if 'username' in data:
        update_fields['username'] = data['username']
    if 'email' in data:
        update_fields['email'] = data['email']
    if 'password' in data:
        update_fields['password'] = generate_password_hash(data['password'])
    result = mongo.db.users.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': update_fields}
    )
    if result.modified_count:
        if 'username' in update_fields:
            session['username'] = update_fields['username']
        log_activity("API Profile Updated", f"User {session['username']} updated profile via API.")
        return jsonify({"message": "Profile updated successfully."}), 200
    else:
        return jsonify({"message": "No changes made."}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use 5000 as a fallback
    app.run(host="0.0.0.0", port=port)

