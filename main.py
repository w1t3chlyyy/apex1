"""
Сервисный Telegram-бот-агент: обработка вебхуков Telegram Business
КОНКРЕТНОГО бота пользователя, RAG-ответы через Qwen, перехват диалогов
человеком.

ОБНОВЛЕНО: вместо Gemini используется Qwen (Alibaba Cloud DashScope,
OpenAI-совместимый режим) — см. qwen_client.py / config.py.

ВАЖНО: этот сервис обслуживает БОТОВ-АГЕНТОВ пользователей (bot_id из URL
/webhook/business/{bot_id} соответствует строке в таблице `bots`, привязанной
к owner_id пользователя личного кабинета). Он НЕ имеет отношения к сервисному
боту авторизации/админ-панели (TELEGRAM_SERVICE_BOT_TOKEN) — тот обслуживается
в Next.js (app/api/bot/webhook/route.ts) и используется для входа в кабинет,
уведомлений владельцу (notify_owner ниже) и админ-панели (рассылки, тарифы).
"""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
import rag
import qwen_client
from supabase import create_client

app = FastAPI(title="Telegram Service Bot")

_supabase = None

# Список стандартных слов-приветствий, на которые ИИ отвечает без проверки базы знаний
GREETINGS = {
    "привет", "здравствуйте", "добрый день", "добрый вечер", 
    "доброе утро", "хай", "хеллоу", "hello", "hi", "салам"
}


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


async def tg_call(token: str, method: str, payload: dict):
    url = TELEGRAM_API.format(token=token, method=method)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload)
        if resp.is_error:
            print(f"❌ Telegram API Error: {resp.status_code} — {resp.text}")
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
    """Пуш-уведомление владельцу с кнопкой «Ответить», когда ИИ не уверен в ответе."""
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

    # Если подписка владельца истекла — бот не отвечает автоматически, а
    # сразу эскалирует диалог на владельца с пометкой об истёкшем тарифе.
    subscription_expires_at = bot_row.get("subscription_expires_at")
    subscription_active = True
    if subscription_expires_at:
        from datetime import datetime, timezone
        try:
            expires = datetime.fromisoformat(subscription_expires_at.replace("Z", "+00:00"))
            subscription_active = expires > datetime.now(timezone.utc)
        except ValueError:
            subscription_active = True

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

    if conv.get("status") == "human_takeover":
        print(f"ℹ️ Диалог {conv['id']} находится под управлением человека. ИИ пропускает сообщение.")
        return JSONResponse({"ok": True})

    if not subscription_active:
        print(f"⚠️ У владельца бота истекла подписка. Перевод диалога {conv['id']} на оператора.")
        supabase.table("conversations").update({"status": "awaiting_human"}).eq("id", conv["id"]).execute()
        if owner_id:
            await notify_owner(owner_id, username, f"[Подписка истекла] {text}", conv["id"])
        return JSONResponse({"ok": True})

    # Поиск контекста в базе знаний
    context, similarity = rag.search_knowledge_base(bot_id, text)

    clean_text = text.strip().lower()
    is_greeting = clean_text in GREETINGS

    print(f"🔍 [RAG DEBUG] Текст: '{text}' | Similarity: {similarity:.4f} | Threshold: {threshold:.4f} | Is Greeting: {is_greeting}")

    # Отвечаем, если порог схожести пройден ИЛИ сообщение является простым приветствием
    if (similarity > threshold and context) or is_greeting:
        print("✅ Условие пройдено! Генерация ответа через Qwen...")
        
        # Если это приветствие и базы знаний нет, передаем подсказку для генерации
        effective_context = context if context else "Клиент поздоровался. Поздоровайся вежливо в ответ и спроси, чем можешь помочь."
        reply = qwen_client.generate_reply(system_prompt, effective_context, text)
        
        print("📤 Отправка сообщения клиенту через Telegram Business...")
        await send_business_message(bot_token, business_connection_id, chat_id, reply)
        
        supabase.table("messages").insert({
            "conversation_id": conv["id"], "role": "assistant", "content": reply,
        }).execute()
    else:
        print(f"❌ Схожесть {similarity:.4f} ниже порога {threshold:.4f}. Эскалация на человека.")
        supabase.table("conversations").update({"status": "awaiting_human"}).eq("id", conv["id"]).execute()
        if owner_id:
            PENDING_REPLIES.setdefault(owner_id, {})
            await notify_owner(owner_id, username, text, conv["id"])

    return JSONResponse({"ok": True})


@app.post("/webhook/service")
async def service_webhook(request: Request):
    supabase = get_supabase()
    update = await request.json()

    callback = update.get("callback_query")
    if callback:
        data = callback["data"]
        owner_id = callback["from"]["id"]
        if data.startswith("reply:"):
            conversation_id = data[len("reply:"):]
            PENDING_REPLIES[owner_id] = {"conversation_id": conversation_id}
            await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "sendMessage", {
                "chat_id": owner_id,
                "text": "Введите ответ для клиента одним сообщением:",
            })
        await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "answerCallbackQuery", {
            "callback_query_id": callback["id"],
        })
        return JSONResponse({"ok": True})

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
    import supabase as supabase_pkg
    url = config.SUPABASE_URL
    key = config.SUPABASE_SERVICE_ROLE_KEY
    return {
        "supabase_url_set": bool(url),
        "supabase_url_preview": url[:20] + "..." if url else None,
        "service_key_set": bool(key),
        "service_key_length": len(key) if key else 0,
        "service_key_prefix": key[:12] + "..." if key else None,
        "supabase_package_version": getattr(supabase_pkg, "__version__", "unknown"),
    }
