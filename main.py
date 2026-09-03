"""
Сервисный Telegram-бот-агент: обработка вебхуков Telegram Business
КОНКРЕТНОГО бота пользователя, RAG-ответы через Qwen, перехват диалогов
человеком.

ОБНОВЛЕНО: вместо Gemini используется Qwen (Alibaba Cloud DashScope,
OpenAI-совместимый режим) — см. qwen_client.py / config.py.

ОБНОВЛЕНО (эскалация): бот всегда пытается ответить сам (используя контекст
базы знаний, если он релевантен, иначе — общие знания), а эскалация на
человека решается моделью через маркер ESCALATE в её ответе (см.
qwen_client.generate_reply) и происходит только в исключительных случаях.

ОБНОВЛЕНО (защита от утечки маркера): раньше эскалация определялась строгим
сравнением reply == "ESCALATE", и если модель добавляла к маркеру хоть
что-то ещё (например "Хорошо, ESCALATE" или обычный ответ с этим словом
внутри) — сравнение не срабатывало, и клиент видел слово ESCALATE прямо в
чате. Теперь is_escalation() ищет маркер ESCALATE как отдельное слово в
ЛЮБОМ месте ответа: если он найден — диалог эскалируется, а сырой текст с
маркером клиенту никогда не отправляется.

ОБНОВЛЕНО (приветствие): раньше бот здоровался в каждом ответе, т.к. каждый
вызов модели не знал, было ли уже приветствие в этом диалоге. Теперь перед
вызовом qwen_client.generate_reply() проверяется, есть ли в диалоге уже
хотя бы одно сообщение от ассистента — если да, модели явно запрещается
здороваться повторно (см. is_first_message ниже и qwen_client.py).

ВАЖНО: этот сервис обслуживает БОТОВ-АГЕНТОВ пользователей (bot_id из URL
/webhook/business/{bot_id} соответствует строке в таблице `bots`, привязанной
к owner_id пользователя личного кабинета). Он НЕ имеет отношения к сервисному
боту авторизации/админ-панели (TELEGRAM_SERVICE_BOT_TOKEN) — тот обслуживается
в Next.js (app/api/bot/webhook/route.ts) и используется для входа в кабинет,
уведомлений владельцу (notify_owner ниже) и админ-панели (рассылки, тарифы).
"""

import re

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config
import rag
import qwen_client
from supabase import create_client

app = FastAPI(title="Telegram Service Bot")

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

# Маркер, которым модель сигнализирует, что нужно передать диалог человеку
# (см. qwen_client.generate_reply). Ищем как отдельное слово в любом месте
# ответа (не строгое равенство!) — это защищает от ситуации, когда модель
# не идеально следует инструкции и добавляет к маркеру лишний текст: такой
# ответ ВСЁ РАВНО будет распознан как эскалация и никогда не уйдёт клиенту.
ESCALATE_PATTERN = re.compile(r"\bESCALATE\b", re.IGNORECASE)


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


def is_escalation(reply: str) -> bool:
    """
    Определяет, попросила ли модель передать диалог человеку. Намеренно НЕ
    сравнивает строку целиком (reply == "ESCALATE"), а ищет маркер как
    отдельное слово где угодно в ответе — иначе при малейшем отклонении
    модели от формата (лишний текст рядом с маркером) слово ESCALATE
    утекало бы прямо в чат клиенту вместо того, чтобы вызвать эскалацию.
    """
    return bool(ESCALATE_PATTERN.search(reply))


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
    text = (biz_message.get("text") or "").strip()

    # Клиент мог прислать стикер/фото/голосовое/видео без текста — Telegram
    # в таком случае не кладёт поле "text" вовсе. Раньше пустая строка
    # улетала прямиком в Qwen embeddings и роняла весь запрос 500-й ошибкой
    # ("input.texts should not be null"). Теперь для отображения владельцу
    # и логов используем понятную подпись, а в RAG/эмбеддинги пустой текст
    # вообще не отправляется (см. rag.py: search_knowledge_base).
    display_text = text if text else "[не-текстовое сообщение: фото/стикер/голосовое/видео и т.п.]"

    bot_row = supabase.table("bots").select("*").eq("id", bot_id).single().execute().data
    if not bot_row:
        return JSONResponse({"ok": True})

    bot_token = bot_row["bot_api_token"]
    owner_id = bot_row["owner_telegram_id"]
    system_prompt = bot_row.get("system_prompt") or "Ты — полезный ассистент поддержки клиентов."

    # ПРИМЕЧАНИЕ: confidence_threshold больше не используется для решения об
    # эскалации — эскалация теперь определяется моделью (см. is_escalation
    # выше и qwen_client.generate_reply). Поле оставлено в схеме/настройках
    # для обратной совместимости, но на поведение бота не влияет.

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
        "conversation_id": conv["id"], "role": "customer", "content": display_text,
    }).execute()

    if conv.get("status") == "human_takeover":
        return JSONResponse({"ok": True})

    if not subscription_active:
        supabase.table("conversations").update({"status": "awaiting_human"}).eq("id", conv["id"]).execute()
        if owner_id:
            await notify_owner(owner_id, username, f"[Подписка истекла] {display_text}", conv["id"])
        return JSONResponse({"ok": True})

    # Не-текстовые сообщения сразу эскалируем на владельца — ИИ по ним
    # ничего не найдёт в базе знаний (search_knowledge_base вернёт "" при
    # пустом тексте), поэтому нет смысла лишний раз дёргать Qwen/RAG.
    if not text:
        supabase.table("conversations").update({"status": "awaiting_human"}).eq("id", conv["id"]).execute()
        if owner_id:
            await notify_owner(owner_id, username, display_text, conv["id"])
        return JSONResponse({"ok": True})

    context, similarity = rag.search_knowledge_base(bot_id, text)

    # Первое ли это сообщение клиента в диалоге? Нужно, чтобы не здороваться
    # заново в каждом ответе (см. qwen_client.generate_reply).
    prior_assistant_msg = (
        supabase.table("messages")
        .select("id")
        .eq("conversation_id", conv["id"])
        .eq("role", "assistant")
        .limit(1)
        .execute()
        .data
    )
    is_first_message = not prior_assistant_msg

    # Бот всегда пытается ответить сам — используя контекст базы знаний как
    # приоритетный источник фактов, а при его отсутствии/нерелевантности
    # отвечая общими знаниями. Эскалация происходит только если модель
    # вернула служебный маркер ESCALATE (см. is_escalation выше).
    reply = qwen_client.generate_reply(system_prompt, context, text, is_first_message)

    if is_escalation(reply):
        supabase.table("conversations").update({"status": "awaiting_human"}).eq("id", conv["id"]).execute()
        PENDING_REPLIES.setdefault(owner_id, {})
        await notify_owner(owner_id, username, text, conv["id"])
    else:
        await send_business_message(bot_token, business_connection_id, chat_id, reply)
        supabase.table("messages").insert({
            "conversation_id": conv["id"], "role": "assistant", "content": reply,
        }).execute()

    return JSONResponse({"ok": True})


# ⚠️ НЕ РЕГИСТРИРОВАТЬ КАК WEBHOOK — см. пояснение в предыдущей версии файла.
# Логика владелец-отвечает-клиенту и админ-панель теперь живут в Next.js
# (app/api/bot/webhook/route.ts), т.к. у сервисного бота может быть только
# один webhook URL.
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
