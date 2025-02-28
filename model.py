import hashlib
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
from attack_detector import AttackClassifier  # Multi-class classifier
from detect_tiny import RegressionCNN           # FGSM regression detector

# Set device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():
    """
    Loads the attack classifier and regression detector models,
    moves them to the proper device, and sets them to evaluation mode.
    """
    classifier = AttackClassifier().to(device)
    classifier.load_state_dict(torch.load("attack_classifier_tiny.pth", map_location=device))
    classifier.eval()
    
    regression_model = RegressionCNN().to(device)
    regression_model.load_state_dict(torch.load("regression_detector_tiny.pth", map_location=device))
    regression_model.eval()
    
    return classifier, regression_model

# Load the models once at module import (outside any function)
classifier, regression_model = load_models()

# Define the transformation to be applied to input images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
])

def compute_sha256(image_bytes):
    """
    Compute the SHA-256 hash of the image file bytes.
    """
    hasher = hashlib.sha256()
    hasher.update(image_bytes)
    return hasher.hexdigest()

def classify_image(file_obj):
    """
    Classify the uploaded image using the loaded models.
    
    Parameters:
      - file_obj: A file-like object (from Flask request.files['image']).
    
    The function:
      1. Reads the image into memory (without saving to disk).
      2. Computes its hash for integrity verification.
      3. Applies the transformation and passes it through the classifier.
      4. Maps the predicted class to a label.
      5. If the attack type is 'FGSM', also obtains a regression output.
      6. Returns the classification result as a string.
    """
    try:
        # Read the file content into memory
        image_bytes = file_obj.read()
        file_obj.seek(0)  # Reset the pointer to allow re-reading
        # Compute original hash before processing
        original_hash = compute_sha256(image_bytes)
        # Load the image from bytes
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logging.error("Error processing image: " + str(e))
        return "Error: Invalid image file"

    # Prepare the image tensor
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

    # Compute hash after processing to verify integrity
    processed_hash = compute_sha256(image_bytes)
    if original_hash != processed_hash:
        return "Error: Image integrity compromised"

    logging.info(f"Predicted attack: {result_text}")
    return result_text
