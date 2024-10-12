from django.shortcuts import render,redirect
from django.core.files.storage import FileSystemStorage
from processing.run import run_on_image
from processing.add_training import launch_training
from processing.database import draw_from_db
# import cv2 
import os
from PIL import Image
import numpy as np
from notebook.settings import MEDIA_ROOT
import shutil
import time

def starting_screen(request):

    data = []
    mask = ''
    if request.method == 'POST' :
    # and request.FILES.get('photo'):
        action = request.POST.get('action')
        if action == 'upload' and request.FILES.get('photo'):
            fs = FileSystemStorage()

            photo = request.FILES['photo']
            serial_num = request.POST.get('serial_num')
            file_extension = os.path.splitext(photo.name)[1]

            filename = fs.save(f'{serial_num}{file_extension}', photo) # имя файла

            # uploaded_file_url = fs.url(filename) # /media/...
            absolute_file_path = os.path.join(fs.location, filename) #fullpath

            # Открываем изображение с помощью Pillow
            image = Image.open(absolute_file_path).convert('RGB')

            # Конвертируем изображение в массив NumPy
            image_array = np.array(image)

            # Получение абсолютного пути к файлу

            mask, process = run_on_image(serial_num,image_array)
            
            context = {
                'uploaded_file_url': process,
                'data': data,
            }
            rnd = render(request, 'starting_screen.html', context)
            
            if process:
                destination_dir = os.path.join(MEDIA_ROOT, 'training_set')
                os.makedirs(destination_dir, exist_ok=True)  # Создаем директорию, если она не существует
                # Перемещаем файл
                src_path = os.path.join(MEDIA_ROOT, f'mask_{serial_num}.png')
                dest_path = os.path.join(destination_dir, f'masks/mask_{serial_num}.png')
                shutil.move(src_path, dest_path)

                src_path = os.path.join(MEDIA_ROOT, filename)
                dest_path = os.path.join(destination_dir, f'data/{filename}')
                shutil.move(src_path, dest_path)
  
            return rnd
        
        elif action == 'relearn':
            time.sleep(5)
            # Переобучение модели
            """
                Разработка происходит на ноутбуке,
                Переобучение модели работает, но на слабом железе может привести к деградации памяти и падению сервера

                Раскомментировать при использовании подходящего оборудования
                launch_training(os.path.join(MEDIA_ROOT, 'training_set')) 
            """
        elif action =='stat':
            draw_from_db()
            return redirect('stat')  
        
    context = {
        'uploaded_file_url': '',
        'data': data,
    }
    return render(request, 'starting_screen.html', context)# Create your views here.

def statistics(request):
    filenm = 'statistics.png'
    return render(request, 'stat.html', {'image_url': filenm})