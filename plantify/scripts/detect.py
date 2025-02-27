import numpy as np
import cv2
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import LabelEncoder
import os
                            
model = load_model('../models/detection_model.keras')
label_encoder = LabelEncoder()
label_encoder.fit(["plant", "non_pant"])
# Load and preprocess the test image
test_image = cv2.imread("C:/Users/Peterven/Desktop/images (2).jpg")
if test_image is None:
    print("Error: Image not found or unable to load.")
else:
    test_image = cv2.resize(test_image, (128, 128))  # Resize image to 128x128
    test_image = np.expand_dims(test_image, axis=0) / 255.0  # Normalize the image
    
    # Predict the label
    predicted_probs = model.predict(test_image)
    predicted_class = np.argmax(predicted_probs, axis=1)
    
    # Decode the label back to original class name
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Predicted Label: {predicted_label[0]}")

    # print(predicted_class)
    # print(np.max(predicted_probs))