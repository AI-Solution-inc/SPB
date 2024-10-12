from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from processing.run import run_on_image
# import cv2 
import os
from PIL import Image
import numpy as np
from notebook.settings import MEDIA_ROOT

def processing() -> list:
    return list()

def starting_screen(request):

    data = []

    if request.method == 'POST' and request.FILES.get('photo'):
        photo = request.FILES['photo']
        fs = FileSystemStorage()
        filename = fs.save(photo.name, photo) # имя файла
        # uploaded_file_url = fs.url(filename) # /media/...
        absolute_file_path = os.path.join(fs.location, filename) #fullpath

        # Открываем изображение с помощью Pillow
        image = Image.open(absolute_file_path).convert('RGB')

        # Конвертируем изображение в массив NumPy
        image_array = np.array(image)

        # Получение абсолютного пути к файлу

        run_on_image(image_array)
        img = os.path.join('media', 'processed.png')
        print(img)

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
