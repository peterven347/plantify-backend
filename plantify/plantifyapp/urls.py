from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('save_plant/', views.save_plant, name='save_plant'),
    path('save_bulk_plants/', views.save_bulk_plants, name='save_bulk_plants'),
    path('get_plant/', views.get_plant, name='get_plant'),
    path('upload_image/', views.upload_image, name='upload_image'),
    path('line/', views.line, name='line'),
]
