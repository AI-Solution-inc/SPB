# Telegram Bot
Телеграм-бот представляет полный прототип решения: достаточно отправить кадр, ответ будет прислан также изображением.

## Инструкция по разворачиванию:
1. Склонируйте репозиторий
2. Создайте в директории `tg-bot/` файл `.env`, запишите в него строку `TOKEN='<YOUR TOKEN>'`, где в поле `YOUR_TOKEN` подставьте токен своего Телеграм-бота
3. Запуск из директории `tg-bot/` 
```python
python -m pip install -r requirements.txt
python main.py
```

# Web интерфейс 
В веб интерфейсе происходит обработка фотографий и визуализация результата
## Инструкция по разворачиванию:
1. Склонируйте репозиторий
```python
curl -sSL https://install.python-poetry.org | python3 -
poetry install 
cd atomic_web
poetry run python manage.py runserver
```
альтернатива 
```python
cd atomic_web
python -m pip install -r requirements.txt
python manage.py runserver
```