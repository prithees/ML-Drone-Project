import tensorflow as tf
import numpy as np
# import RPi.GPIO as GPIO
import time
from PIL import Image, ImageOps
import requests
import os
import keyboard

FLASK_SERVER_URL = "http://127.0.0.1:5000/get_class_name"

def get_class_name():
    try:
        response = requests.get(FLASK_SERVER_URL)
        response.raise_for_status()
        data = response.json()
        return data["class_name"]
    except requests.RequestException as e:
        print(f"Error retrieving class name: {e}")
        return None

# Get the class name from the Flask server
class1 = get_class_name()
print("Class Name:", class1)

LED_PIN = 14
# GPIO.setmode(GPIO.BCM)  
# GPIO.setup(LED_PIN, GPIO.OUT)

# Load the model and class names
model = tf.keras.models.load_model("Uploads/keras_model.h5", compile=False)
class_names = [line.strip() for line in open("Uploads/labels.txt", "r").readlines()]

@tf.function(reduce_retracing=True)
def predict_image(image_array):
    normalized_image_array = (tf.cast(image_array, tf.float32) / 127.5) - 1
    data = tf.expand_dims(normalized_image_array, axis=0)
    prediction = model(data)
    index = tf.argmax(prediction, axis=-1)
    confidence_score = tf.reduce_max(prediction)
    return index, confidence_score

image_folder = "uploads/"  

i = 1
while True:
    img_name = f"img{i}"  
    full_image_path = os.path.join(image_folder, f"{img_name}.png")  
    
    if keyboard.is_pressed('enter'):
        print("Enter pressed, exiting the loop.")
        break
    
    if not os.path.exists(full_image_path):
        print(f"No more images found for {img_name}. Exiting.")
        break
    
    try:
        image = Image.open(full_image_path).convert("RGB")
        
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        image_array = np.asarray(image)

        if image_array.shape != (224, 224, 3):
            raise ValueError(f"Expected image shape (224, 224, 3), got {image_array.shape}")

        index_tensor, confidence_tensor = predict_image(image_array)
        index = index_tensor.numpy()[0]
        confidence_score = confidence_tensor.numpy()

        class_name1 = class_names[index]

        print(f"Predicted Class for {img_name}: {class_name1}")
        print(f"Confidence Score: {confidence_score}")
        if class_name1 != class1:
            print(f"No Need to be Sprayed for {img_name}")
        else:
            for j in range(1):  
                print(f"Water Sprayed for {img_name}")
                time.sleep(2)
                # GPIO.output(LED_PIN, GPIO.HIGH)  # Uncomment when using GPIO
                print("Stopped")
                time.sleep(2)
            print(f"Need to be Sprayed for {img_name}")

    except Exception as e:
        print(f"Error processing {full_image_path}: {e}")
    
    i += 1 