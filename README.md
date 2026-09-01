# Telegram Service Bot

FastAPI-сервис, который:

1. Принимает вебхуки **Telegram Business** от ботов клиентов
   (`POST /webhook/business/{bot_id}`).
2. Ищет ответ в базе знаний клиента через **pgvector** (Supabase).
3. Если сходство > `RAG_CONFIDENCE_THRESHOLD` — отвечает от лица бизнес-аккаунта
   через **Gemini API**.
4. Если сходство ниже порога — отправляет владельцу push-уведомление в
   служебный бот с inline-кнопкой **«Ответить»**.
5. При нажатии владельцем «Ответить» и вводе текста — пересылает ответ
   напрямую покупателю через Telegram Business API.

## Запуск

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # заполните переменные
uvicorn bot:app --host 0.0.0.0 --port 8000
```

## Установка вебхуков

Для служебного бота (уведомления/команды владельцев):

```bash
curl -X POST "https://api.telegram.org/bot<SERVICE_BOT_TOKEN>/setWebhook" \
     -d "url=${PUBLIC_BASE_URL}/webhook/service"
```

Для каждого бота клиента, подключённого к Telegram Business, вебхук
устанавливается автоматически при сохранении токена в личном кабинете
(см. `register_client_bot_webhook` в `bot.py`), либо вручную:

```bash
curl -X POST "https://api.telegram.org/bot<CLIENT_BOT_TOKEN>/setWebhook" \
     -d "url=${PUBLIC_BASE_URL}/webhook/business/<bot_id>"
```

## Структура

- `bot.py` — FastAPI-приложение, роуты вебхуков, логика перехвата диалогов
- `rag.py` — поиск по базе знаний (pgvector) через Supabase RPC
- `gemini_client.py` — обёртка над Gemini API (chat + embeddings)
- `config.py` — загрузка переменных окружения
