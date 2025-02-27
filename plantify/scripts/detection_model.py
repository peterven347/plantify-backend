import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load images and labels from a directory
def load_data_from_directory(image_dir):
    images = []
    labels = []
    
    # for label in os.listdir(image_dir):

        # label_dir = os.path.join(image_dir, label)
        # if os.path.isdir(label_dir):
        #     for root, dirs, files in os.walk(label_dir):
        #         for file in files:
        #             print(file)
        #             image_path = os.path.join(label_dir, file)
        #             image = cv2.imread(image_path)
        #             if image is not None:
        #                 image = cv2.resize(image, (128, 128))  # Resize image to 128x128
        #                 images.append(image)
        #                 labels.append(label)
    for label in os.listdir(image_dir):
        label_dir = os.path.join(image_dir, label)
        if os.path.isdir(label_dir):
            for filename in os.listdir(label_dir):
                if os.path.isdir(os.path.join(label_dir, filename)) == False:
                    image_path = os.path.join(label_dir, filename)
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, (128, 128))  # Resize image to 128x128
                        images.append(image)
                        labels.append(label)
                else:
                    full_path = os.path.join(label_dir, filename)
                    if os.path.isdir(full_path):
                        for filename in os.listdir(full_path):
                            image_path = os.path.join(full_path, filename)
                            image = cv2.imread(image_path)
                            if image is not None:
                                image = cv2.resize(image, (128, 128))  # Resize image to 128x128
                                images.append(image)
                                labels.append(label)
    return images, labels

# Load your dataset (replace this with your dataset directory)
base_path = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
# image_dir = '../data/train'  # Path where images are stored
image_dir = "C:/Users/Peterven/CWP/plantify/plantify-backEnd/plantify/data/train"
# image_dir = os.path.join(base_path, "../data")
images, labels = load_data_from_directory(image_dir)

# Convert images to numpy array and normalize the pixel values
images = np.array(images) / 255.0

# Encode labels into numeric values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# One-hot encode the labels for multi-class classification
y = to_categorical(encoded_labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# Build the CNN model using Keras
model = Sequential()

# Add the first convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add third convolutional layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output from the previous layer
model.add(Flatten())

# Add a fully connected (dense) layer
model.add(Dense(512, activation='relu'))

# Add dropout for regularization
model.add(Dropout(0.5))

# Output layer: number of classes (e.g., number of plant species)
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model
os.makedirs('../models/', exist_ok=True)
model.save('../models/detection_model.keras')
print("Model saved successfully!")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Optionally plot training/validation graphs
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy/Loss')
plt.legend()
plt.show()