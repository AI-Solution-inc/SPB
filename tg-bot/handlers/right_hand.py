from aiogram.exceptions import TelegramBadRequest
from aiogram import Bot, types
from aiogram import Router
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import Command
from io import BytesIO
import os
import numpy as np
from PIL import Image

from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

import matplotlib
matplotlib.use('Agg')

router = Router()

model = YOLO("D:\\DOCS\\contest\\atomic_hack\\DetectDefects\\tg-bot\\model\\best (1).pt")
colors = ['#19F754', '#F7F019', '#F77919', '#F73E19', '#510808']

@router.message(Command('start'))
async def start(message: Message):
    await message.answer('Пожалуйста, загрузите картинку со сварочными швами.')


@router.message(Command('help'))
async def start(message: Message):
    await message.answer('Этот бот поможет найти дефекты на картинках со сварочными швами. Пожалуйста, загрузите картинку.')


@router.message(Command('info'))
async def start(message: Message, bot: Bot):
    curr_path = os.path.dirname(os.path.realpath(__file__))
    await bot.send_message(message.from_user.id, "Цветовые обозначения:")
    with BytesIO() as output:
        with Image.open(os.path.join(curr_path, "info.png")) as img:
            img.save(output, 'PNG')
        raw = output.getvalue()
    file2send = BufferedInputFile(raw, "result.png")
    await bot.send_photo(message.from_user.id, file2send)


@router.message(Command("clear"))
async def all_clear(message: Message, bot: Bot):
    try:
        for i in range(message.message_id, 0, -1):
            await bot.delete_message(message.from_user.id, i)
    except TelegramBadRequest as ex:

        if ex.message == "Bad Request: message to delete not found":
            print("Все сообщения удалены")


@router.message()
async def photo_handler(message: types.Message,  bot: Bot):
    if message.content_type == 'photo':
        file = await bot.get_file(message.photo[-1].file_id)
        file_path = file.file_path
        data: BytesIO = await bot.download_file(file_path)
        img = np.array(Image.open(data))
        await bot.send_message(message.from_user.id, "Фото загружено, идет обработка!")

        pred = model(img, imgsz=640, device='cpu')[0]

        boxes = pred.boxes.xywhn.numpy()
        classes = pred.boxes.cls.numpy().tolist()

        await bot.send_message(message.from_user.id, f"Найдено {len(boxes)} объектов, идет отрисовка")

        height, width, _ = img.shape

        fig, ax1 = plt.subplots(figsize=(width / 100, height / 100), dpi=100, facecolor='none')
        ax1.imshow(img)
        ax1.axis('off')

        for bb, cls in zip(boxes, classes):
            clDe, xCen, yCen, widBB, heiBB = (int(cls), float(bb[0]) * width, float(bb[1]) * height,
                                            float(bb[2]) * width, float(bb[3]) * height)
            xLeBB, yLeBB = xCen - (widBB / 2), yCen - (heiBB / 2)

            rectTig = patches.Rectangle((xLeBB, yLeBB), widBB, heiBB, linewidth=3, edgecolor=colors[clDe], facecolor='none')
            ax1.add_patch(rectTig)

        plt.subplots_adjust(0,0,1,1,0,0)
        fig.canvas.draw()
        image_proc = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_proc = image_proc.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        with BytesIO() as output:
            with Image.fromarray(image_proc) as img:
                img.save(output, 'PNG')
            raw = output.getvalue()
        file2send = BufferedInputFile(raw, "result.png")
        
        await bot.send_photo(message.from_user.id, file2send)