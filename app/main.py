import asyncio
import json
import time
import re
import os
from contextlib import asynccontextmanager
from typing import List, Optional, Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

from web_selectors import (
    INPUT_SELECTOR,
    INPUT_SELECTORS_LIST,
    SEND_BUTTON_SELECTOR,
    SEND_BUTTONS_LIST,
    NEW_CHAT_SELECTORS,
    ASSISTANT_MSG_SELECTOR,
    PROSE_SELECTOR,
    STOP_BUTTON_SELECTORS,
)

load_dotenv()

CHROME_PATH = os.getenv("CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
CDP_URL = os.getenv("CDP_URL", "http://127.0.0.1:9333")
CHAT_URL = os.getenv("CHAT_URL", "https://gemma4.com/chat")
USER_DATA_DIR = os.getenv("USER_DATA_DIR", r"C:\temp\chrome_debug")


# --- Менеджер состояния браузера (Singleton) ---

class BrowserManager:
    """Управляет жизненным циклом Playwright и экземпляра браузера Chrome."""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.is_initialized = False
        self.lock = asyncio.Lock()

    async def check_chrome_available(self) -> bool:
        """Проверяет, отвечает ли Chrome по порту 9333 через прямое TCP-подключение"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection('127.0.0.1', 9333),
                timeout=2.0
            )
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            return True
        except (asyncio.TimeoutError, ConnectionRefusedError) as e:
            print(f"🔍 Порт 9333 недоступен: {type(e).__name__}")
            return False
        except Exception as e:
            print(f"🔍 Непредвиденная ошибка при проверке порта 9333: {e}")
            return False

    async def get_page(self):
        """Подключается к запущенному браузеру и возвращает рабочую страницу."""
        if self.is_initialized and self.browser:
            try:
                await self.page.evaluate("window.location.href")
                return self.page
            except (PlaywrightTimeoutError, Exception) as e:
                print(f"⚠️ Страница недоступна, пересоздаем: {e}")
                self.is_initialized = False

        if not await self.check_chrome_available():
            error_msg = (
                "\n" + "="*50 + "\n"
                "❌ ВНИМАНИЕ: Chrome не запущен или недоступен по порту 9333!\n"
                "Пожалуйста, запустите Chrome вручную с помощью вашего ярлыка:\n"
                f'"{CHROME_PATH}" --remote-debugging-port=9333 --user-data-dir="{USER_DATA_DIR}"\n'
                + "="*50 + "\n"
            )
            print(error_msg)
            raise RuntimeError("Chrome not available. Please launch it manually.")

        if not self.playwright:
            self.playwright = await async_playwright().start()

        if not self.browser:
            self.browser = await self.playwright.chromium.connect_over_cdp(CDP_URL)
            print("✅ Подключены к Chrome")

        if not self.context:
            self.context = self.browser.contexts[0] if self.browser.contexts else await self.browser.new_context()

        if not self.page:
            self.page = await self.context.new_page()
        else:
            try:
                await self.page.evaluate("window.location.href")
            except Exception:
                self.page = await self.context.new_page()

        current_url = await self.page.evaluate("window.location.href")
        if CHAT_URL not in current_url:
            print(f"🌐 Открываем {CHAT_URL}...")
            await self.page.goto(CHAT_URL, wait_until="domcontentloaded")
            await self.page.wait_for_load_state("networkidle", timeout=15000)
            await asyncio.sleep(2)

        self.is_initialized = True
        return self.page

    async def close(self):
        """Очищает и закрывает все соединения при завершении работы сервера."""
        print("🧹 Закрытие соединений с браузером...")
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
        self.is_initialized = False


# Инициализация менеджера
browser_manager = BrowserManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Код до yield выполняется при запуске приложения
    yield
    # Код после yield выполняется при остановке Uvicorn/FastAPI
    await browser_manager.close()


app = FastAPI(title="Cline to Web-LLM Bridge", lifespan=lifespan)


# --- Модели данных OpenAI ---

class ToolCallFunction(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolCallFunction

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[Any]] = None 
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]


# --- Вспомогательные функции веб-интерфейса ---

async def start_new_chat(page):
    """Инициализирует новый чат в веб-интерфейсе"""
    print("✨ Создаем новый чат в веб-интерфейсе...")
    try:
        await page.goto(CHAT_URL, wait_until="domcontentloaded")
        await asyncio.sleep(2)
        
        for sel in NEW_CHAT_SELECTORS:
            if await page.locator(sel).count() > 0:
                await page.locator(sel).first.click()
                print(f"✅ Нажата кнопка нового чата: {sel}")
                await asyncio.sleep(1.5)
                break
    except Exception as e:
        print(f"⚠️ Ошибка при создании нового чата: {e}")

async def send_prompt(page, prompt_text: str):
    try:
        input_field = None
        for selector in INPUT_SELECTORS_LIST:
            if await page.locator(selector).count() > 0:
                input_field = page.locator(selector).first
                break
                
        if not input_field:
            print("❌ Поле ввода не найдено")
            return False
        
        await input_field.click()
        await asyncio.sleep(0.2)
        
        try:
            await input_field.fill("")
        except Exception:
            pass
        
        escaped_prompt = json.dumps(prompt_text)
        await page.evaluate(f"""
            (function() {{
                const input = document.querySelector('textarea, [contenteditable="true"]');
                if (!input) return;
                if (input.tagName === 'TEXTAREA' || input.tagName === 'INPUT') {{
                    input.value = {escaped_prompt};
                }} else {{
                    input.innerText = {escaped_prompt};
                }}
                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }})();
        """)
        
        await asyncio.sleep(0.5)
        
        try:
            await input_field.type(" ", delay=10)
            await input_field.press("Backspace")
        except Exception:
            pass
        
        await asyncio.sleep(0.5)
        
        try:
            await input_field.press("Enter")
            await asyncio.sleep(2)
            return True
        except Exception:
            pass
        
        for btn_selector in SEND_BUTTONS_LIST:
            if await page.locator(btn_selector).count() > 0:
                await page.locator(btn_selector).first.click()
                await asyncio.sleep(2)
                return True
                
        return False
    except Exception as e:
        print(f"❌ Ошибка при отправке: {e}")
        return False

async def is_generating(page) -> bool:
    """Проверяет, идёт ли сейчас процесс генерации ответа в UI."""
    for sel in STOP_BUTTON_SELECTORS:
        try:
            if await page.locator(sel).is_visible():
                return True
        except Exception:
            pass
    return False

async def read_response(page, prev_msg_count: int):
    try:
        timeout = 180
        print("⏳ Ожидание появления нового сообщения...")
        message_count = prev_msg_count
        
        for _ in range(120):
            current_count = await page.locator(ASSISTANT_MSG_SELECTOR).count()
            if current_count > prev_msg_count:
                message_count = current_count
                print(f"✅ Появилось новое сообщение! (Всего: {message_count})")
                break
            await asyncio.sleep(1)
            
        if message_count <= prev_msg_count:
            return "❌ Ответ от модели не появился"
            
        assistant_messages = await page.locator(ASSISTANT_MSG_SELECTOR).all()
        last_message = assistant_messages[-1]
        
        print("⏳ Ожидание завершения генерации...")
        last_content = ""
        stable_count = 0
        read_start = time.time()
        
        await asyncio.sleep(1.0)
        
        while (time.time() - read_start) < timeout:
            currently_generating = await is_generating(page)
            
            content = None
            
            try:
                prose_element = await last_message.locator(PROSE_SELECTOR).first
                if await prose_element.count() > 0:
                    content = await prose_element.inner_text()
            except Exception:
                pass
            
            if not content or len(content.strip()) == 0:
                content = await last_message.inner_text()
                content = content.replace("Копировать", "").strip()
                
            if content and len(content.strip()) > 0:
                content = content.strip()
                if content.startswith("Gemma 4"):
                    content = content.replace("Gemma 4", "").strip()
                
                if content != last_content:
                    last_content = content
                    stable_count = 0
                    print(f"⏳ Генерируется... ({len(content)} симв.)")
                else:
                    stable_count += 1
                
                if not currently_generating:
                    extracted = extract_json_from_response(content)
                    
                    try:
                        parsed = json.loads(extracted, strict=False)
                        if isinstance(parsed, dict) and "tool_calls" in parsed:
                            print("✅ Генерация завершена: обнаружена кнопка завершения и валидный tool_calls JSON")
                            return extracted
                    except Exception:
                        pass
                    
                    if stable_count >= 15:
                        print("✅ Генерация завершена (индикатор Stop пропал, текст стабилен 2с)")
                        return extract_json_from_response(content)
                else:
                    # Если мы всё ещё генерируем, проверяем, не является ли JSON «завершенным»
                    # (начинается с { и заканчивается на }, но json.loads еще не смог его распарсить)
                    extracted = extract_json_from_response(content)
                    if extracted and extracted.startswith('{') and extracted.endswith('}'):
                        try:
                            json.loads(extracted, strict=False)
                            # Если распарсилось, значит JSON полон. Но согласно требованию, 
                            # при currently_generating == True мы не должны возвращать контент, 
                            # даже если он стал стабильным, unless stable_count очень велик.
                        except Exception:
                            # JSON не полон, продолжаем ждать
                            pass

                    # Приоритет у currently_generating == False. 
                    # Если индикатор активен, увеличиваем порог стабильности перед сдачей.
                    if stable_count >= 60: # Значительно выше чем 30
                        print("⚠️ Достигнута критическая пауза при активной генерации (30с), возвращаем текущий вывод")
                        return extract_json_from_response(content)
            
            await asyncio.sleep(0.5)
            
        if last_content:
            return extract_json_from_response(last_content)
            
        return "❌ Превышено время ожидания"
    except Exception as e:
        print(f"❌ Ошибка при чтении: {e}")
        return f"❌ Ошибка: {e}"


def extract_json_from_response(content: str) -> str:
    if not content:
        return content

    content = content.replace('\xa0', ' ').replace('\u00a0', ' ').strip()
    content = re.sub(r'^(?:Gemma\s*4|json|copy|\s)+', '', content, flags=re.IGNORECASE).strip()

    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        potential_json = content[start:end+1]
    else:
        potential_json = content

    try:
        parsed = json.loads(potential_json, strict=False)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            return json.dumps(parsed)
    except Exception:
        pass

    def fix_args(match):
        prefix, body, suffix = match.group(1), match.group(2), match.group(3)
        body_clean = body.replace('\\"', '"')
        body_escaped = body_clean.replace('"', '\\"')
        return f'{prefix}{body_escaped}{suffix}'

    fixed_json = re.sub(r'("arguments"\s*:\s*")(\{.*?\})("\s*[\},])', fix_args, potential_json, flags=re.DOTALL)

    try:
        parsed = json.loads(fixed_json, strict=False)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            return json.dumps(parsed)
    except Exception:
        pass

    code_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    for block in code_blocks:
        try:
            parsed = json.loads(block.strip(), strict=False)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                return json.dumps(parsed)
        except Exception:
            continue

    return potential_json


def build_current_turn_prompt(request: ChatCompletionRequest) -> tuple[str, bool]:
    last_assistant_idx = -1
    for i, msg in enumerate(request.messages):
        if msg.role == "assistant":
            last_assistant_idx = i

    is_new_task = (last_assistant_idx == -1)
    prompt = ""

    format_instruction = r"""
When you need to use a tool, respond ONLY with a JSON object inside ```json ... ``` block or as raw JSON.

CRITICAL JSON RULES:
1. The response MUST be a valid JSON object with a "tool_calls" array.
2. Inside "arguments", all double quotes MUST be properly escaped (\").

EXAMPLE OF VALID RESPONSE:
{
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "read_files",
        "arguments": "{\"files\": [{\"path\": \"C:\\\\Projects\\\\main.py\"}]}"
      }
    }
  ]
}

Respond ONLY with JSON. No prose, no conversation, no markdown explanations outside JSON.
пиши ответ в соответствии с форматом OpenAI Tool Use
большие файлы читай частями
"""

    if is_new_task:
        system_prompt = next((msg.content for msg in request.messages if msg.role == "system" and msg.content), "")
        prompt += f"### SYSTEM PROMPT ###\n{system_prompt}\n\n"
        
        if request.tools:
            tools_list = [f"- {t.get('function', {}).get('name')}: {t.get('function', {}).get('description')}" for t in request.tools]
            prompt += f"### AVAILABLE TOOLS ###\n" + "\n".join(tools_list) + "\n\n"
            
        prompt += "### CONVERSATION START ###\n"
        msgs_to_send = [m for m in request.messages if m.role != "system"]
    else:
        prompt += "### NEW INPUTS ###\n"
        msgs_to_send = request.messages[last_assistant_idx + 1:]

    for msg in msgs_to_send:
        if msg.role == "user":
            prompt += f"\n[USER INPUT]:\n{msg.content}\n"
        elif msg.role == "tool":
            prompt += f"\n[TOOL RESULT (from {msg.name})]:\n{msg.content}\n"

    prompt += f"\n\n### FORMAT INSTRUCTION ###\n{format_instruction.strip()}\n"
    prompt += "\n**IMPORTANT:** If using tools, respond with ONLY the JSON. NO extra text."
    return prompt.strip(), is_new_task


async def stream_generator(content: Optional[str], tool_calls_data: Optional[List[Dict[str, Any]]] = None):
    response_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())
    
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gemma-4', 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
    await asyncio.sleep(0.01)
    
    if tool_calls_data:
        yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gemma-4', 'choices': [{'index': 0, 'delta': {'tool_calls': tool_calls_data}, 'finish_reason': None}]})}\n\n"
    elif content:
        if content.strip().startswith('<') and '>' in content:
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gemma-4', 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
        else:
            sentences = re.split(r'([.!?]+\s+)', content)
            chunks = []
            for i in range(0, len(sentences), 2):
                if i + 1 < len(sentences): chunks.append(sentences[i] + sentences[i+1])
                else: chunks.append(sentences[i])
            if len(chunks) <= 1:
                words = content.split()
                chunks = [' '.join(words[i:i+3]) + (' ' if i+3 < len(words) else '') for i in range(0, len(words), 3)]
            
            for chunk in chunks:
                if chunk.strip():
                    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gemma-4', 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                    await asyncio.sleep(0.02)
    
    finish_reason = "tool_calls" if tool_calls_data else "stop"
    yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': 'gemma-4', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
    yield "data: [DONE]\n\n"


# --- Middleware ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"\n📦 {request.method} {request.url}")
    response = await call_next(request)
    return response


# --- Эндпоинты ---

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "gemma-4", "object": "model", "created": int(time.time()), "owned_by": "google"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    async with browser_manager.lock:
        turn_prompt, is_new_task = build_current_turn_prompt(request)
        print(f"\n[{'НОВАЯ ЗАДАЧА' if is_new_task else 'ПРОДОЛЖЕНИЕ ЗАДАЧИ'}] - Длина промпта: {len(turn_prompt)}")
        
        try:
            page = await browser_manager.get_page()
        except (PlaywrightTimeoutError, RuntimeError, Exception) as e:
            print(f"❌ Ошибка инициализации браузера: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Browser execution context unavailable: {str(e)}"
            )
        
        if is_new_task:
            await start_new_chat(page)
            
        try:
            prev_msg_count = await page.locator(ASSISTANT_MSG_SELECTOR).count()
        except (PlaywrightTimeoutError, Exception):
            prev_msg_count = 0
            
        success = await send_prompt(page, turn_prompt)
        if not success:
            print("❌ Не удалось отправить промпт")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to send prompt to the upstream model"
            )
            
        ai_response_text = await read_response(page, prev_msg_count)
        
        if ai_response_text.startswith("❌"):
            print(f"❌ Ошибка получения ответа: {ai_response_text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=ai_response_text
            )
        
        tool_calls_data = None
        response_content = ai_response_text
        finish_reason = "stop"
        
        clean_response_text = extract_json_from_response(ai_response_text)
        
        try:
            parsed_json = json.loads(clean_response_text, strict=False)
            # Строгая валидация: считаем ответом tool_calls только если это словарь с ключом tool_calls,
            # который содержит список, и при этом нет основного текстового контента, который мог бы быть основным ответом.
            if isinstance(parsed_json, dict) and "tool_calls" in parsed_json and isinstance(parsed_json["tool_calls"], list):
                tool_calls_data = parsed_json["tool_calls"]
                response_content = None
                finish_reason = "tool_calls"
                print("🔧 Распарсили tool_calls для Cline")
            else:
                response_content = ai_response_text
        except (json.JSONDecodeError, TypeError, ValueError):
            response_content = ai_response_text
        
        if request.stream:
            return StreamingResponse(
                stream_generator(response_content, tool_calls_data),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            return ChatCompletionResponse(
                id=f"chatcmpl-bridge-{int(time.time())}",
                created=int(time.time()),
                model="gemma-4",
                choices=[ChatCompletionResponseChoice(
                    index=0, 
                    message=Message(role="assistant", content=response_content, tool_calls=tool_calls_data), 
                    finish_reason=finish_reason
                )]
            )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
