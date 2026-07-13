from flask import Flask, request, render_template, jsonify
import os
import threading
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create the uploads directory if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load the ML model and class names
model = tf.keras.models.load_model("uploads/keras_model.h5", compile=False)
with open("uploads/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"Loaded {len(class_names)} class names.")

# Global variable to store the last predicted class name
predicted_class_name = None
predicted_class_lock = threading.Lock()

@tf.function(reduce_retracing=True)
def predict_image(image_array):
    normalized_image_array = (tf.cast(image_array, tf.float32) / 127.5) - 1
    data = tf.expand_dims(normalized_image_array, axis=0)
    prediction = model(data)
    index = tf.argmax(prediction, axis=-1)
    confidence_score = tf.reduce_max(prediction)
    return index, confidence_score

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
    global predicted_class_name  # Declare global to update the value
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if not allowed_file(file.filename):
        return "Unsupported file type. Please upload a PNG or JPEG image.", 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded image with the ML model
        image = Image.open(file_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)

        if image_array.shape != (224, 224, 3):
            return f"Expected image shape (224, 224, 3), got {image_array.shape}"
        
        index_tensor, confidence_tensor = predict_image(image_array)
        index = index_tensor.numpy()[0]
        confidence_score = confidence_tensor.numpy()

        print(f"Predicted index: {index}, Confidence score: {confidence_score}")

        # Update the global variable with the latest class name
        with predicted_class_lock:
            predicted_class_name = class_names[index]
        return render_template('result.html', class_name=predicted_class_name, confidence_score=confidence_score)

@app.route("/get_class_name", methods=['GET'])
def get_class_name():
    with predicted_class_lock:
        class_name = predicted_class_name
    if class_name is None:
        return jsonify({"error": "No class name available. Please upload an image first."}), 404
    return jsonify({"class_name": class_name})

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
