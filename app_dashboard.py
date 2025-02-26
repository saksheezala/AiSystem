import hashlib
from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import logging
from attack_detector import AttackClassifier  # Multi-class classifier
from detect_tiny import RegressionCNN         # FGSM regression detector
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

def compute_sha256(image_path):
    """Compute SHA-256 hash of an image file."""
    hasher = hashlib.sha256()
    with open(image_path, 'rb') as img_file:
        hasher.update(img_file.read())
    return hasher.hexdigest()

def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = AttackClassifier().to(device)
    classifier.load_state_dict(torch.load("attack_classifier_tiny.pth", map_location=device))
    classifier.eval()
    
    regression_model = RegressionCNN().to(device)
    regression_model.load_state_dict(torch.load("regression_detector_tiny.pth", map_location=device))
    regression_model.eval()
    
    return classifier, regression_model, device

classifier, regression_model, device = load_models()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975),
                         (0.2770, 0.2691, 0.2821))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    original_hash = compute_sha256(file_path)  # Compute hash before processing

    try:
        img = Image.open(file_path).convert("RGB")
    except Exception as e:
        logging.error("Error processing image: " + str(e))
        return jsonify({'error': 'Invalid image file'}), 400

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        class_output = classifier(img_tensor)
        pred_class = class_output.argmax(dim=1).item()
        class_labels = {0: "Normal", 1: "FGSM", 2: "PGD"}
        attack_type = class_labels.get(pred_class, "Unknown")
        
        if attack_type == "FGSM":
            epsilon_pred = regression_model(img_tensor).item()
            result_text = f"Attack Type: FGSM"
        else:
            result_text = f"Attack Type: {attack_type}"
    
    # Verify integrity after processing
    processed_hash = compute_sha256(file_path)
    if original_hash != processed_hash:
        return jsonify({'error': 'Image integrity compromised. Possible tampering detected.'}), 400
    
    logging.info(f"Predicted attack: {result_text}")
    if attack_type != "Normal":
        socketio.emit('alert', {'message': f'Adversarial attack detected: {attack_type}'})
    
    return jsonify({'result': result_text})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port)