import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.metrics import classification_report


def get_data_generators(data_dir, image_size=(224, 224), batch_size=32, validation_split=0.2):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split 
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=validation_split)

    train_dir = os.path.join(data_dir, 'train/plant')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

base_path = os.path.dirname(os.path.abspath(__file__))  # Current directory of the script
data_dir = os.path.join(base_path, "../data")
image_size = (224, 224)
batch_size = 32
num_epochs = 10

train_generator, validation_generator = get_data_generators(
    data_dir=data_dir,
    image_size=image_size,
    batch_size=batch_size
)

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs
)

base_model.trainable = True
fine_tune_at = len(base_model.layers) // 2

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 5
total_epochs = num_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1]
)

os.makedirs('../models/', exist_ok=True)
model.save('../models/classification_model.keras')
print("Model saved successfully!")



predictions = model.predict(train_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = train_generator.classes
class_labels = list(train_generator.class_indices.keys())

report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)
