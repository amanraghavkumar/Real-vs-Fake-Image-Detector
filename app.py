
from flask import Flask, request, render_template, redirect, url_for
import io, base64
import numpy as np
from PIL import Image
import gdown
import os
from tensorflow.keras.models import load_model


template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)


# Config
MODEL_PATH = "model/best_model.h5"
DRIVE_URL = "https://drive.google.com/file/d/1prG4ByWvUISmE2iaYX9ao76CClOtcMAG/view?usp=sharing"

IMAGE_SIZE = (256, 256)
LABELS = ("Fake","Real")  # Folder order pe depend karta hai

# Flask app
app = Flask(__name__)

# Download if not exists
if not os.path.exists(MODEL_PATH):
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)


# Load model
model = load_model(MODEL_PATH)

# ---------- PREPROCESS FUNCTION ----------
def preprocess_image(img_bytes, target_size=IMAGE_SIZE):
    # RGB image load karo
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(target_size)

    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dim add

    return arr, img

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_data = None

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(url_for("index"))

        f = request.files["file"]
        if f.filename == "":
            return redirect(url_for("index"))

        img_bytes = f.read()
        x, pil_img = preprocess_image(img_bytes)

        preds = model.predict(x)
        # sigmoid output
        prob = float(preds[0][0])
        label_idx = 1 if prob >= 0.5 else 0
        result = LABELS[label_idx]
        confidence = f"{prob:.3f}" if label_idx == 1 else f"{1-prob:.3f}"

        # image preview
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template("index.html", result=result, confidence=confidence, img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True)
