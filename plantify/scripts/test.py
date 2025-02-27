import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
import os

img_path = "C:/Users/Peterven/Desktop/1735321173240.jpg"

detection_model = load_model('../models/detection_model.keras')
classification_model = load_model("../models/classification_model.keras")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "../data/train/plant",  # Replace with your test directory "../data/train"
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # or 'binary' depending on your classification
)
label_encoder = LabelEncoder()
label_encoder.fit(["plant", "non_pant"])
test_image = cv2.imread(img_path)

if test_image is None:
    print("Error: Image not found or unable to load.")
else:
    test_image = cv2.resize(test_image, (128, 128))  # Resize image to 128x128
    test_image = np.expand_dims(test_image, axis=0) / 255.0  # Normalize the image
    
    # Predict the label
    predicted_probs = detection_model.predict(test_image)
    predicted_class = np.argmax(predicted_probs, axis=1)
    
    # Decode the label back to original class name
    predicted_label = label_encoder.inverse_transform(predicted_class)
    # print(f"Predicted Label: {predicted_label[0]}")
    # print(predicted_class)
    # print(np.max(predicted_probs))


    if predicted_label[0] == "plant":
        img = image.load_img(img_path, target_size=(224, 224))  # Resizing to the same size as the training images
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0                             # Normalize the image

        predictions = classification_model.predict(img_array)        # Predict the class
        predicted_class = np.argmax(predictions, axis=1)
        fraction = np.max(predictions)
        print(fraction)

        class_labels = {v: k for k, v in test_generator.class_indices.items()}
        # print(f"Predicted class: {predicted_class}")
        print(class_labels[predicted_class[0]])



