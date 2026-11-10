# Telegram Service Bot

FastAPI-сервис, который:

1. Принимает вебхуки **Telegram Business** от ботов клиентов
   (`POST /webhook/business/{bot_id}`).
2. Ищет ответ в базе знаний клиента через **pgvector** (Supabase).
3. Если сходство > `RAG_CONFIDENCE_THRESHOLD` — отвечает от лица бизнес-аккаунта
   через Qwen API.
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
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Установка вебхуков

Для служебного бота (вход, уведомления и команды владельцев) webhook должен
указывать на Next.js-приложение `apex`, где реализован обработчик
`/api/bot/webhook`. Не направляйте служебного бота на этот FastAPI-сервис:
здесь такого маршрута нет. Установите URL фронтенда и тот же секрет, что задан
как `TELEGRAM_WEBHOOK_SECRET` в Vercel:

```bash
curl -X POST "https://api.telegram.org/bot<SERVICE_BOT_TOKEN>/setWebhook" \
     -d "url=${APEX_FRONTEND_URL}/api/bot/webhook" \
     -d "secret_token=${TELEGRAM_WEBHOOK_SECRET}" \
     -d 'allowed_updates=["message","callback_query"]'
```

Для каждого бота клиента, подключённого к Telegram Business, вебхук
устанавливается автоматически при сохранении токена в личном кабинете
(см. `app/api/bot/telegram-token/route.ts` в репозитории `apex`), либо вручную:

```bash
curl -X POST "https://api.telegram.org/bot<CLIENT_BOT_TOKEN>/setWebhook" \
     -d "url=${PUBLIC_BASE_URL}/webhook/business/<bot_id>"
```

`PUBLIC_BASE_URL` — адрес именно этого FastAPI-сервиса; он используется для
вебхуков клиентских Business-ботов, а не служебного бота.

## Структура

- `main.py` — FastAPI-приложение, роуты вебхуков, логика перехвата диалогов
- `rag.py` — поиск по базе знаний (pgvector) через Supabase RPC
- `qwen_client.py` — обёртка над Qwen API (chat + embeddings)
- `config.py` — загрузка переменных окружения
