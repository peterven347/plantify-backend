import numpy as np
import cv2
import os
import time
import math
import openpyxl
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, throttle_classes
from rest_framework.throttling import AnonRateThrottle, UserRateThrottle
from rest_framework.exceptions import Throttled
from sklearn.preprocessing import LabelEncoder
from .models import PlantModel
from mongoengine import DoesNotExist
from .serializers import PlantSerializer
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from scripts.excel_extract import get_plants


base_path = os.path.dirname(os.path.abspath(__file__))  # Current directory of this file
detection_model = load_model(os.path.join(base_path, "../models/detection_model.keras"))
classification_model = load_model(os.path.join(base_path, "../models/classification_model.keras"))

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(base_path, "../data/train/plant"), # test directory
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
# steps = test_generator.samples
# if steps == 0:
#     steps == 1
# print ("steps", steps)

label_encoder = LabelEncoder()
label_encoder.fit(["plant", "non_plant"])


sheet_path = os.path.join(base_path,"../../plants_ext_for_db.xlsx")
workbook = openpyxl.load_workbook(sheet_path)
sheet = workbook["Original"]
start_row = 2
end_row = 54


def predict_image(img_path):
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
        # print(np.max(predicted_probs))

        if predicted_label[0] == "plant" and np.max(predicted_probs) > 0.8:
            img = image.load_img(img_path, target_size=(224, 224))  # Resizing to the same size as the training images
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array /= 255.0                             # Normalize the image

            predictions = classification_model.predict(img_array)        # Predict the class
            predicted_class = np.argmax(predictions, axis=1)
            fraction = np.max(predictions)
            if fraction > 0.49:
                class_labels = {v: k for k, v in test_generator.class_indices.items()}
                return(f"{math.floor(fraction * 100)}% {class_labels[predicted_class[0]]}")
            else:
                print(fraction)
                return("no plant record to retrieve")
        else:
            return("no plant image detected")

def home(request):
    return HttpResponse("Hello, World!")

@csrf_exempt
@api_view(['POST'])
# @throttle_classes([AnonRateThrottle, UserRateThrottle])  # Applying throttling to this view
# @throttle_classes([UserRateThrottle])  # Applying throttling to this view
def upload_image(request):
    if request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg'] # Check for allowed file types
        if uploaded_file.content_type not in allowed_types:
            return JsonResponse({"message": "Invalid file format"}, status=400)

        timestamp_filename = str(int(time.time())) + '-' + uploaded_file.name
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        file_path = fs.save(timestamp_filename, uploaded_file)
        file_path = os.path.join(base_path, "../uploads/images/", file_path)
        plant = predict_image(file_path)
        os.remove(file_path)

        if plant == "no plant image detected":
            return JsonResponse({"message": "no plant image detected"})
        elif plant == "no plant record to retrieve":
            return JsonResponse({"message": "no plant record to retrieve"})
        else:
            try:
                plantdb = PlantModel.objects.get(scientific_name = plant.split("% ")[1])
                plant_data = plantdb.to_mongo()
                plant_data["_id"] = str(plant_data["_id"])
                return JsonResponse({"percentage": plant.split("% ")[0], "plant_data": plant_data})
            except DoesNotExist:
                return JsonResponse({"message": "no plant record to retrieve"})
            except Exception as e:
                print(f"Error: {e}")
                return JsonResponse({"message": "an error occured"})
    return JsonResponse({"status": False, "message": "No file uploaded"}, status=400)

@csrf_exempt
@api_view(['PUT'])
def save_plant(request):
    plant = PlantModel(
        # family =
        scientific_name = "test"
        # common_name = 
        # local_names = 
        # medicinal_use = []
    #     # hausa_name = 
    #     # igbo_name = 
    #     # yoruba_name = 
    )
    print(plant)
    plant.save()
    return JsonResponse({"message": "plant saved"})


@csrf_exempt
@api_view(['PUT'])
def save_bulk_plants(request):
    plants = []
    class PlantObj:
        def __init__(self, family="", scientific_name="", common_name="", yoruba_name="", igbo_name="", hausa_name="", plant_part="", medicinal_use=""):
            self.family = family
            self.scientific_name = scientific_name
            self.common_name = common_name
            self.yoruba_name = yoruba_name
            self.igbo_name = igbo_name
            self.hausa_name = hausa_name
            self.plant_part = plant_part
            self.medicinal_use = medicinal_use
        def __str__(self):
            # This method returns a formatted string representation of the plant object
            return f"{{\n\tfamily: '{self.family}',\n\tscientific_name: '{self.scientific_name}'," \
                f"\n\tcommon_name: '{self.common_name}',\n\tyoruba_name: '{self.yoruba_name}',\n\tigbo_name: '{self.igbo_name}',\n\thausa_name: '{self.hausa_name}'," \
                f"\n\tplant_part: '{self.plant_part}'\n\tmedicinal_use: '{self.medicinal_use}'\n}}"
        
    def get_plants():
        for row_number in range(start_row, end_row + 1):
            row = sheet[row_number]
            family = row[0].value.strip()
            scientific_name = str(row[1].value).strip()
            common_name = str(row[2].value).strip()
            yoruba_name = str(row[3].value).strip()
            igbo_name = str(row[4].value).strip()
            hausa_name = str(row[5].value).strip()
            plant_part = str(row[6].value).strip()
            medicinal_use = str(row[7].value).strip()

            plant = PlantObj(family, scientific_name, common_name, yoruba_name, igbo_name, hausa_name, plant_part, medicinal_use)
            plants.append(plant)
        for i in plants:
            saveplant = PlantModel(
                family = i.family,
                scientific_name = i.scientific_name,
                common_name = i.common_name,
                yoruba_name = i.yoruba_name,
                igbo_name = i.igbo_name,
                hausa_name = i.hausa_name,
                plant_part = i.plant_part,
                medicinal_use = i.medicinal_use
            )
            saveplant.save()
    try:
        get_plants()
        return JsonResponse({"message": "selected plants saved"}, status=200)
    except Exception as e:
        print(e)



@api_view(['GET'])
def get_plant(request):
    # print(f"Request user: {request.user}")
    plant = PlantModel.objects.get(scientific_name = "Bryophyllum pinnatum (Lam) Oken")
    plant_data = plant.to_mongo()
    plant_data["_id"] = str(plant_data["_id"])
    return JsonResponse(plant_data)

@api_view(['GET'])
@throttle_classes([UserRateThrottle])
def line(request):
    try:
        return JsonResponse({"message": "line hit"}, status=200)
    except Exception as e:
        return JsonResponse({"message": "error"}, status=429)
