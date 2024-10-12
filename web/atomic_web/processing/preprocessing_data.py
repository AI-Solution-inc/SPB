# processing/preprocessing_functions.py
from ultralytics import YOLO 
import cv2 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import matplotlib
matplotlib.use('Agg') 

# def preprocess_data_json(file_path):

#     # Реализация функции предобработки данных
#     colors = ['#19F754', '#F7F019', '#F77919', '#F73E19', '#510808']    

#     imgT = cv2.imread(file_path)

#     pred = modelDef(imgT, imgsz = 640)[0]

#     boxes = pred.boxes.xywhn
#     cls = pred.boxes.cls

#     coordinates = boxes.tolist()
#     classes = cls.tolist()

#     fig = plt.figure(figsize = (8, 10))
#     fig,ax1 = fig.add_subplot()
#     ax1.imshow(imgT)
#     ax1.set_title(f'{listW[ind]}')
#     width, height = imgT.shape[1],  imgT.shape[0]
#     for i in range(len(boxes)):
#         bb = boxes[i]
#         clDe, xCen, yCen, widBB, heiBB = (int(cls[i]), float(bb[0]) * width, float(bb[1]) * height, 
#                                         float(bb[2]) * width, float(bb[3]) * height)
#         #plt.scatter(xCen, yCen, s = 12, c = colors[clDef])
#         xLeBB, yLeBB = xCen - (widBB / 2),  yCen - (heiBB / 2)

#         rectTig = patches.Rectangle((xLeBB, yLeBB), widBB, heiBB, linewidth=1.5, edgecolor= colors[clDe], facecolor='none')
#         ax1.add_patch(rectTig)
        
#     plt.show()
#     # Создание списка словарей в нужном формате
#     processed_data = []
#     for coord, cls in zip(coordinates, classes):
#         processed_data.append({
#             "class": int(cls),
#             "x": coord[0],
#             "y": coord[1],
#             "width": coord[2],
#             "height": coord[3]
#         })

#     return processed_data


modelDef = YOLO('processing/best (1).pt')

def preprocess_data_json(file_path):
    # Реализация функции предобработки данных
    colors = ['#19F754', '#F7F019', '#F77919', '#F73E19', '#510808']    

    imgT = cv2.imread(file_path)
    pred = modelDef(imgT, imgsz=640)[0]

    boxes = pred.boxes.xywhn
    cls = pred.boxes.cls

    coordinates = boxes.tolist()
    classes = cls.tolist()

    height, width, _ = imgT.shape
    fig, ax1 = plt.subplots(figsize=(width / 100, height / 100), dpi=100, facecolor='none')
    ax1.imshow(imgT)
    ax1.axis('off')
    width, height = imgT.shape[1], imgT.shape[0]
    for i in range(len(boxes)):
        bb = boxes[i]
        clDe, xCen, yCen, widBB, heiBB = (int(cls[i]), float(bb[0]) * width, float(bb[1]) * height,
                                          float(bb[2]) * width, float(bb[3]) * height)
        xLeBB, yLeBB = xCen - (widBB / 2), yCen - (heiBB / 2)

        rectTig = patches.Rectangle((xLeBB, yLeBB), widBB, heiBB, linewidth=5, edgecolor=colors[clDe], facecolor='none')
        ax1.add_patch(rectTig)

    # Сохранение изображения в файл
    output_image_path = os.path.join('media', 'output_image.png')  # Путь к папке media
    plt.savefig(output_image_path, transparent=True, pad_inches=0)
    plt.close(fig)

    # Создание списка словарей в нужном формате
    processed_data = []
    for coord, cls in zip(coordinates, classes):
        processed_data.append({
            "class": int(cls),
            "x": coord[0],
            "y": coord[1],
            "width": coord[2],
            "height": coord[3]
        })

    # Возвращаем путь к сохраненному изображению
    return processed_data, output_image_path
