import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}




def get_resnet18():
    """Build ResNet-18 model for binary DR classification."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model


def get_efficientnet_b0():
    """Build EfficientNet-B0 model for binary DR classification."""
    model = models.efficientnet_b0(weights='DEFAULT')
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)
    return model


def load_single_model(arch, model_path, device):
    """Load weights for a given architecture. Returns (model, loaded_bool)."""
    if arch == "resnet18":
        model = get_resnet18().to(device)
    else:
        model = get_efficientnet_b0().to(device)

    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)

        msg = model.load_state_dict(state, strict=False)

        print(f"{arch} load message:", msg)
        model.eval()
        return model, True
    model.eval()
    return model, False


# ── Global state ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "resnet18":        {"path": "models/dr_model.pth",          "instance": None, "loaded": False},
    "efficientnet_b0": {"path": "models/efficientnet_b0.pth",   "instance": None, "loaded": False},
}

def get_or_load(arch):
    """Lazy-load and cache the requested model."""
    entry = MODELS[arch]
    if entry["instance"] is None:
        m, loaded = load_single_model(arch, entry["path"], device)
        entry["instance"] = m
        entry["loaded"] = loaded
    return entry["instance"], entry["loaded"]

# Pre-load both at startup
for _arch in MODELS:
    get_or_load(_arch)

# Report startup status
for _arch, _entry in MODELS.items():
    print(f"[{_arch}] loaded={_entry['loaded']}, device={device}")


# Preprocessing pipeline matching training
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor, image





def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload PNG, JPG, JPEG, BMP, or TIFF"}), 400

    # Determine which model to use (form field or query param, default resnet18)
    arch = request.form.get("model", request.args.get("model", "resnet18")).strip().lower()
    if arch not in MODELS:
        arch = "resnet18"

    try:
        image_bytes = file.read()
        tensor, pil_image = preprocess_image(image_bytes)
        tensor = tensor.to(device)

        model, model_loaded = get_or_load(arch)

        with torch.no_grad():
            output = model(tensor)

        if arch == "efficientnet_b0":
            probs = torch.softmax(output, dim=1)
            prob = probs[0][1].item()   # DR probability
            pred = torch.argmax(probs, dim=1).item()

        else:
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0
    
        confidence = prob if pred == 1 else (1 - prob)

        return jsonify({
            "success": True,
            "model_loaded": model_loaded,
            "architecture": arch,
            "prediction": "Diabetic Retinopathy" if pred == 1 else "Normal Eye",
            "confidence": round(confidence * 100, 2),
            "dr_probability": round(prob * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models": {arch: entry["loaded"] for arch, entry in MODELS.items()},
        "device": str(device)
    })


if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    app.run(debug=True, port=5000)