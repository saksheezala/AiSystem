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
    classifier = AttackClassifier().to(device)
    classifier.load_state_dict(torch.load("attack_classifier_tiny.pth", map_location=device))
    classifier.eval()
    
    regression_model = RegressionCNN().to(device)
    regression_model.load_state_dict(torch.load("regression_detector_tiny.pth", map_location=device))
    regression_model.eval()
    
    # Move models to CPU if GPU is used
    classifier.to("cpu")
    regression_model.to("cpu")

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
    try:
        # Load models inside function (if not frequent use)
        classifier, regression_model = load_models()
        
        # Process image (same steps as before)
        image_bytes_io = io.BytesIO(file_obj.read())
        file_obj.seek(0)
        img = Image.open(image_bytes_io).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Inference
        with torch.inference_mode():
            class_output = classifier(img_tensor)
            pred_class = class_output.argmax(dim=1).item()
            attack_type = {0: "Normal", 1: "FGSM", 2: "PGD"}.get(pred_class, "Unknown")

            if attack_type == "FGSM":
                epsilon_pred = regression_model(img_tensor).item()
                result_text = f"Attack Type: FGSM (ε ≈ {epsilon_pred:.4f})"
            else:
                result_text = f"Attack Type: {attack_type}"

        # Free memory
        del classifier, regression_model
        torch.cuda.empty_cache()  # If using GPU

        return result_text
    except Exception as e:
        return f"Error: {e}"

