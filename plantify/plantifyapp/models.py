import datetime
from django.db import models
from mongoengine import Document, StringField, DateTimeField, ListField

class PlantModel(Document):
    scientific_name = StringField(required=True, unique=True, max_length=200)
    common_name = StringField()
    family = StringField()
    genus = StringField()
    order = StringField()
    local_names = StringField()
    hausa_name = StringField()
    igbo_name = StringField()
    yoruba_name = StringField()
    plant_part = StringField()
    medicinal_use = StringField()
    meta = {
        'collection': 'plant_data',
        'indexes': [
            {'fields': ['scientific_name'], 'unique': True}  # Ensure uniqueness at the database level
        ]
    }