"""
Сервисный Telegram-бот-агент: обработка вебхуков Telegram Business
КОНКРЕТНОГО бота пользователя, RAG-ответы через Gemini, перехват диалогов
человеком.

ВАЖНО: этот сервис обслуживает БОТОВ-АГЕНТОВ пользователей (bot_id из URL
/webhook/business/{bot_id} соответствует строке в таблице `bots`, привязанной
к owner_id пользователя личного кабинета). Он НЕ имеет отношения к сервисному
боту авторизации (TELEGRAM_SERVICE_BOT_TOKEN) — тот обслуживается в Next.js
(app/api/bot/webhook/route.ts) и используется только для входа в кабинет и
для уведомлений владельцу (notify_owner ниже).

ИСПРАВЛЕНИЕ: раньше при первом сообщении клиента business_connection_id не
сохранялся в таблицу conversations, из-за чего при передаче диалога человеку
(service_webhook -> send_business_message) ответ владельца не мог быть
отправлен клиенту (conv.get("business_connection_id") возвращал None).
Теперь business_connection_id сохраняется в upsert ниже.
"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
import rag
import gemini_client
from supabase import create_client

app = FastAPI(title="Telegram Service Bot")

# Ленивая инициализация: раньше клиент создавался здесь же на уровне модуля,
# из-за чего при отсутствии/неверных SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY
# падал ИМПОРТ всего модуля (и, соответственно, весь процесс — даже /health
# переставал отвечать). Теперь клиент создаётся при первом реальном
# обращении, а при отсутствии ключей выбрасывается понятная ошибка вместо
# крипто-трейса "Invalid API key" на старте.
_supabase = None


def get_supabase():
    global _supabase
    if _supabase is None:
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
            raise RuntimeError(
                "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY не заданы в переменных окружения"
            )
        _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
    return _supabase


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"

# Простое in-memory состояние "владелец сейчас отвечает клиенту X" (для прод — использовать Redis)
PENDING_REPLIES: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Вспомогательные функции Telegram API
# ---------------------------------------------------------------------------

async def tg_call(token: str, method: str, payload: dict):
    url = TELEGRAM_API.format(token=token, method=method)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_business_message(bot_token: str, business_connection_id: str, chat_id: int, text: str):
    """Отправляет сообщение от лица подключённого Telegram Business аккаунта."""
    return await tg_call(bot_token, "sendMessage", {
        "business_connection_id": business_connection_id,
        "chat_id": chat_id,
        "text": text,
    })


async def notify_owner(owner_telegram_id: int, customer_username: str, question: str, conversation_id: str):
    """Пуш-уведомление владельцу с кнопкой «Ответить», когда ИИ не уверен в ответе.

    Уведомление шлётся через СЕРВИСНЫЙ бот авторизации (TELEGRAM_SERVICE_BOT_TOKEN),
    т.к. владелец логинится в личный кабинет именно через него — это ожидаемо
    и НЕ является той же ошибкой, что была с ботами-агентами: здесь как раз
    нужен именно сервисный бот, чтобы у владельца был единый чат для всех
    уведомлений от любого количества его агентов.
    """
    text = f"⚠️ ИИ не знает ответа на вопрос «{question}» от @{customer_username or 'unknown'}"
    keyboard = {
        "inline_keyboard": [[
            {"text": "Ответить", "callback_data": f"reply:{conversation_id}"}
        ]]
    }
    await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "sendMessage", {
        "chat_id": owner_telegram_id,
        "text": text,
        "reply_markup": keyboard,
    })


# ---------------------------------------------------------------------------
# Вебхук от Telegram Business (сообщения клиентов конкретному боту-агенту)
# ---------------------------------------------------------------------------

@app.post("/webhook/business/{bot_id}")
async def business_webhook(bot_id: str, request: Request):
    supabase = get_supabase()

    update = await request.json()
    biz_message = update.get("business_message")
    if not biz_message:
        return JSONResponse({"ok": True})

    business_connection_id = biz_message.get("business_connection_id")
    chat = biz_message["chat"]
    chat_id = chat["id"]
    username = chat.get("username", "")
    text = biz_message.get("text", "")

    bot_row = supabase.table("bots").select("*").eq("id", bot_id).single().execute().data
    if not bot_row:
        return JSONResponse({"ok": True})

    bot_token = bot_row["bot_api_token"]
    owner_id = bot_row["owner_telegram_id"]
    system_prompt = bot_row.get("system_prompt") or "Ты — полезный ассистент поддержки клиентов."
    threshold = bot_row.get("confidence_threshold") or config.RAG_CONFIDENCE_THRESHOLD

    # Сохраняем входящее сообщение. business_connection_id сохраняем ВСЕГДА
    # (в т.ч. обновляем при каждом сообщении) — он нужен позже, чтобы
    # владелец мог ответить клиенту через send_business_message.
    conv = supabase.table("conversations").upsert(
        {
            "bot_id": bot_id,
            "customer_chat_id": chat_id,
            "customer_username": username,
            "business_connection_id": business_connection_id,
        },
        on_conflict="bot_id,customer_chat_id",
    ).execute().data[0]

    supabase.table("messages").insert({
        "conversation_id": conv["id"], "role": "customer", "content": text,
    }).execute()

    # Если диалог уже перехвачен человеком — не отвечаем автоматически
    if conv.get("status") == "human_takeover":
        return JSONResponse({"ok": True})

    context, similarity = rag.search_knowledge_base(bot_id, text)

    if similarity > threshold and context:
        reply = gemini_client.generate_reply(system_prompt, context, text)
        await send_business_message(bot_token, business_connection_id, chat_id, reply)
        supabase.table("messages").insert({
            "conversation_id": conv["id"], "role": "assistant", "content": reply,
        }).execute()
    else:
        supabase.table("conversations").update({"status": "awaiting_human"}).eq("id", conv["id"]).execute()
        PENDING_REPLIES.setdefault(owner_id, {})
        await notify_owner(owner_id, username, text, conv["id"])

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# ⚠️ НЕ РЕГИСТРИРОВАТЬ КАК WEBHOOK. Оставлено только для справки/локального
# тестирования логики. У сервисного бота может быть только ОДИН webhook URL,
# и это теперь app/api/bot/webhook/route.ts в Next.js — там реализована
# ТА ЖЕ логика (обработка callback_query "reply:" и текста-ответа владельца),
# см. handleOwnerCallback / handleOwnerReplyText в этом файле Next.js.
# Если вы предпочитаете держать эту логику здесь (в Python), а не в Next.js —
# удалите её дублирование из route.ts и зарегистрируйте вебхук сервисного
# бота именно на этот роут вместо Next.js. Делать оба одновременно нельзя.
# ---------------------------------------------------------------------------

@app.post("/webhook/service")
async def service_webhook(request: Request):
    supabase = get_supabase()

    update = await request.json()

    # Нажатие кнопки "Ответить"
    callback = update.get("callback_query")
    if callback:
        data = callback["data"]
        owner_id = callback["from"]["id"]
        if data.startswith("reply:"):
            conversation_id = data.split(":", 1)[1]
            PENDING_REPLIES[owner_id] = {"conversation_id": conversation_id}
            await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "sendMessage", {
                "chat_id": owner_id,
                "text": "Введите ответ для клиента одним сообщением:",
            })
        await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "answerCallbackQuery", {
            "callback_query_id": callback["id"],
        })
        return JSONResponse({"ok": True})

    # Текстовое сообщение владельца — если ожидается ответ клиенту, пересылаем его
    message = update.get("message")
    if message and "text" in message:
        owner_id = message["from"]["id"]
        pending = PENDING_REPLIES.get(owner_id)
        if pending:
            conversation_id = pending["conversation_id"]
            conv = supabase.table("conversations").select("*, bots(*)").eq("id", conversation_id).single().execute().data
            bot_row = conv["bots"]

            await send_business_message(
                bot_row["bot_api_token"],
                conv.get("business_connection_id"),
                conv["customer_chat_id"],
                message["text"],
            )

            supabase.table("messages").insert({
                "conversation_id": conversation_id, "role": "owner", "content": message["text"],
            }).execute()
            supabase.table("conversations").update({"status": "human_takeover"}).eq("id", conversation_id).execute()

            del PENDING_REPLIES[owner_id]
            await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "sendMessage", {
                "chat_id": owner_id, "text": "Ответ отправлен клиенту ✅",
            })

    return JSONResponse({"ok": True})


@app.get("/health")
@app.get("/debug/env")
async def debug_env():
    url = config.SUPABASE_URL
    key = config.SUPABASE_SERVICE_ROLE_KEY
    return {
        "supabase_url_set": bool(url),
        "supabase_url_preview": url[:20] + "..." if url else None,
        "service_key_set": bool(key),
        "service_key_length": len(key) if key else 0,
        "service_key_prefix": key[:12] + "..." if key else None,
    }
async def health():
    return {"status": "ok"}
