from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

model = load_model("C:/Users/Peterven/CWP/plantify/plantify-backEnd/plantify/models/classification_model.keras")
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "C:/Users/Peterven/CWP/plantify/plantify-backEnd/plantify/data/train/plant",  # Replace with your test directory "../data/train"
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'  # or 'binary' depending on your classification
)

# # Print the number of samples in the test set
# print(f"Number of test samples: {test_generator.samples}")

# # Calculate steps per epoch and print it
# steps = test_generator.samples // test_generator.batch_size
# if steps == 0:
#     steps = 1  # Ensure steps is at least 1

# print(f"Steps per epoch: {steps}")
# test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
# print(f"Test Loss: {test_loss}")
# print(f"Test Accuracy: {test_accuracy}")


def evaluate_model():

    # Print the number of samples in the test set
    print(f"Number of test samples: {test_generator.samples}")

    # Calculate steps per epoch and print it
    steps = test_generator.samples // test_generator.batch_size
    if steps == 0:
        steps = 1  # Ensure steps is at least 1

    print(f"Steps per epoch: {steps}")

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy


def predict_image(img_path):
    # img_path = 'C:/Users/Peterven/Desktop/class diagram_011550.png'
    img = image.load_img(img_path, target_size=(224, 224))  # Resizing to the same size as the training images
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Normalize the image
    img_array /= 255.0

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    nn = np.max(predictions)
    print(nn)


    class_labels = {v: k for k, v in test_generator.class_indices.items()}
    print(f"Predicted class: {predicted_class}")
    return class_labels[predicted_class[0]]

print(predict_image('C:/Users/Peterven/Desktop/images (2).jpg'))
# evaluate_model()