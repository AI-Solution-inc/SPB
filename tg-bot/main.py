import asyncio
import logging

from aiogram import Bot, Dispatcher
from keyboards.menu import default_commands
from handlers import right_hand
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
bot = Bot(token=os.getenv("TOKEN"))

dp = Dispatcher()
dp.include_router(right_hand.router)

dp.startup.register(default_commands)
dp.run_polling(bot)


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())