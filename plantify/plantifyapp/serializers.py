# serializers.py in plantifyapp
from rest_framework import serializers
from .models import PlantModel

class PlantSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlantModel
        fields = '__all__'
