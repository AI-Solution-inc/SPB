from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from processing.preprocessing_data import preprocess_data_json
import cv2 
import os

def processing() -> list:
    return list()

def starting_screen(request):

    data = []

    if request.method == 'POST' and request.FILES.get('photo'):
        photo = request.FILES['photo']
        fs = FileSystemStorage()
        filename = fs.save(photo.name, photo)
        uploaded_file_url = fs.url(filename)

        # Получение абсолютного пути к файлу
        absolute_file_path = os.path.join(fs.location, filename)

        data, img  = preprocess_data_json(absolute_file_path)

        context = {
            'uploaded_file_url': img,
            'data': data,
        }
        return render(request, 'starting_screen.html', context)
    
    context = {
        'uploaded_file_url': '',
        'data': data,
    }

    return render(request, 'starting_screen.html', context)# Create your views here.
