from aiogram import types


# определим кнопки на всякий случай
class InlineKeyboard:
    @property
    def start(self):

        inline_keyboard = [
            [types.InlineKeyboardButton(text='Some options', callback_data='options'),
             types.InlineKeyboardButton(text="Download", callback_data="download"),]
        ]
        return types.InlineKeyboardMarkup(inline_keyboard=inline_keyboard)