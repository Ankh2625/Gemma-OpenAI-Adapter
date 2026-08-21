# Селекторы для интерфейса чата gemma4.com

# Поле ввода сообщения
INPUT_SELECTOR = 'div[contenteditable="true"]'
INPUT_SELECTORS_LIST = [
    "textarea#prompt-textarea",
    "textarea[placeholder]",
    "textarea",
    "div[contenteditable='true']",
    "div[role='textbox']",
]

# Кнопка и селекторы отправки
SEND_BUTTON_SELECTOR = 'button[aria-label="Send message"], button.send-button'
SEND_BUTTONS_LIST = [
    'button[aria-label="Send message"]',
    'button.send-button',
    'button[type="submit"]',
]

# Кнопки для создания нового чата
NEW_CHAT_SELECTORS = [
    "a[href='/chat']", 
    "button:has-text('New chat')",
    "button:has-text('Новый чат')",
    "button[aria-label='New chat']",
    "button[title='New chat']",
    ".new-chat-btn"
]

# Сообщения ассистента (для подсчета и чтения)
ASSISTANT_MSG_SELECTOR = 'div.msg-assistant'

# Контейнер с текстом внутри сообщения ассистента
PROSE_SELECTOR = '.prose'

# Кнопка остановки генерации
STOP_BUTTON_SELECTORS = [
    'button[aria-label="Stop generating"]',
    'button[aria-label="Stop"]',
    "button:has-text('Stop')",
    "button:has-text('Остановить')",
    "button.stop-button",
]