import tensorflow as tf
import numpy as np
import RPi.GPIO as GPIO
import time
from PIL import Image, ImageOps

LED_PIN = 14
GPIO.setmode(GPIO.BCM)  
GPIO.setup(LED_PIN, GPIO.OUT)

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

image_name = "affe2.png"
image = Image.open(image_name).convert("RGB")

size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)

if image_array.shape != (224, 224, 3):
    raise ValueError(f"Expected image shape (224, 224, 3), got {image_array.shape}")

index_tensor, confidence_tensor = predict_image(image_array)
index = index_tensor.numpy()[0]
confidence_score = confidence_tensor.numpy()

class_name = class_names[index]

print(class_name)
print(confidence_score)

if class_name == "2 turmeric_good":
    import os
    os.remove(image_name)
    print("No Need to be Sprayed")
else:
    for i in range(3):
            GPIO.output(LED_PIN, GPIO.HIGH)
            print("LED ON")
            time.sleep(2)
            GPIO.output(LED_PIN, GPIO.LOW)
            print("LED OFF")
            time.sleep(2)
    print("Need to be Sprayed")
