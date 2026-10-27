"""
Telegram-бот-агент: вебхуки Telegram Business бота КОНКРЕТНОГО пользователя,
RAG-ответы через Qwen, эскалация на человека.

Правила работы:
- Вебхук принимает только запросы с верным X-Telegram-Bot-Api-Secret-Token.
- Бот всегда пытается ответить сам; эскалация — маркер ESCALATE от модели.
- Подписка истекла -> бот молчит, диалог уходит владельцу.
- Бесплатный тариф (нет plan_id): лимит токенов FREE_TIER_TOKEN_LIMIT.
- Платный тариф: лимит «ответов ИИ» за 30 дней из plans.messages_limit.
- Сообщения самого владельца клиенту только логируются.
- human_takeover снимается через HUMAN_TAKEOVER_HOURS после ответа владельца.

Служебный бот (вход, ответ владельца, админ-панель) обслуживается в Next.js
(app/api/bot/webhook/route.ts); здесь он только отправляет уведомления.
"""

import hmac
import re
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

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
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY не заданы")
        _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
    return _supabase


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
USAGE_WINDOW = timedelta(days=30)

# Маркер эскалации ищем как отдельное слово в любом месте ответа: даже если
# модель добавит лишний текст, клиенту он не уйдёт.
ESCALATE_PATTERN = re.compile(r"\bESCALATE\b", re.IGNORECASE)

HEADER_UNKNOWN = "⚠️ ИИ не знает ответа на вопрос"
HEADER_EXPIRED = "⏳ Подписка истекла, ИИ не отвечает. Вопрос клиента:"
HEADER_LIMIT = "📉 Лимит ответов ИИ по тарифу исчерпан, ИИ не отвечает. Вопрос клиента:"
HEADER_ERROR = "⚠️ Не удалось получить ответ ИИ. Вопрос клиента:"


def is_escalation(reply: str) -> bool:
    return bool(ESCALATE_PATTERN.search(reply))


def parse_dt(value):
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def tg_call(token: str, method: str, payload: dict):
    url = TELEGRAM_API.format(token=token, method=method)
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


async def send_business_message(bot_token: str, business_connection_id: str, chat_id: int, text: str):
    return await tg_call(bot_token, "sendMessage", {
        "business_connection_id": business_connection_id,
        "chat_id": chat_id,
        "text": text,
    })


async def notify_owner(owner_telegram_id, customer_username, question, conversation_id, header=HEADER_UNKNOWN):
    """Пуш владельцу с кнопкой «Ответить». Ошибки Telegram не должны ронять вебхук."""
    if not owner_telegram_id:
        return
    text = f"{header} «{question}» от @{customer_username or 'unknown'}"
    keyboard = {"inline_keyboard": [[{"text": "Ответить", "callback_data": f"reply:{conversation_id}"}]]}
    try:
        await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "sendMessage", {
            "chat_id": owner_telegram_id,
            "text": text,
            "reply_markup": keyboard,
        })
    except Exception as exc:
        print(f"[notify_owner] не удалось отправить уведомление: {exc}")


async def notify_owner_limit_reached(owner_telegram_id, token_limit: int):
    if not owner_telegram_id:
        return
    text = (
        f"🚫 Бесплатный лимит в {token_limit} токенов ответов ИИ исчерпан.\n\n"
        "Ваш AI-ассистент больше не отвечает клиентам автоматически — "
        "диалоги теперь нужно вести вручную. Чтобы бот снова заработал, "
        "оформите платный тариф в личном кабинете → раздел «Подписка»."
    )
    try:
        await tg_call(config.TELEGRAM_SERVICE_BOT_TOKEN, "sendMessage", {
            "chat_id": owner_telegram_id,
            "text": text,
        })
    except Exception as exc:
        print(f"[notify_owner_limit_reached] не удалось отправить уведомление: {exc}")


def get_usage(bot_row):
    """(использовано ответов, начало окна). Окно 30 дней; по истечении обнуляется."""
    now = datetime.now(timezone.utc)
    start = parse_dt(bot_row.get("ai_replies_period_start"))
    if start is None or now - start >= USAGE_WINDOW:
        return 0, now
    return int(bot_row.get("ai_replies_used") or 0), start


def get_plan_messages_limit(supabase, plan_id):
    """Лимит ответов за 30 дней. None = без лимита."""
    try:
        rows = supabase.table("plans").select("messages_limit").eq("id", plan_id).limit(1).execute().data
    except Exception as exc:
        print(f"[plans] не удалось прочитать тариф {plan_id}: {exc}")
        rows = None
    if rows:
        return rows[0].get("messages_limit")
    return config.DEFAULT_PLAN_MESSAGE_LIMITS.get(plan_id)


def takeover_expired(supabase, conversation_id) -> bool:
    """Диалог в human_takeover: прошло ли HUMAN_TAKEOVER_HOURS с ответа владельца."""
    if config.HUMAN_TAKEOVER_HOURS <= 0:
        return False
    rows = (
        supabase.table("messages")
        .select("created_at")
        .eq("conversation_id", conversation_id)
        .eq("role", "owner")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    last = parse_dt(rows[0]["created_at"]) if rows else None
    if last is None:
        return True
    return datetime.now(timezone.utc) - last >= timedelta(hours=config.HUMAN_TAKEOVER_HOURS)


def set_status(supabase, conversation_id, status):
    supabase.table("conversations").update({"status": status}).eq("id", conversation_id).execute()


def webhook_secret_ok(request: Request) -> bool:
    expected = config.TELEGRAM_WEBHOOK_SECRET
    if not expected:
        print("[webhook] TELEGRAM_WEBHOOK_SECRET не задан — запросы отклоняются")
        return False
    got = request.headers.get("x-telegram-bot-api-secret-token", "")
    return hmac.compare_digest(got, expected)


@app.post("/webhook/business/{bot_id}")
async def business_webhook(bot_id: str, request: Request):
    if not webhook_secret_ok(request):
        return JSONResponse({"error": "forbidden"}, status_code=403)

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
    display_text = text if text else "[не-текстовое сообщение: фото/стикер/голосовое/видео и т.п.]"

    rows = supabase.table("bots").select("*").eq("id", bot_id).limit(1).execute().data
    if not rows:
        return JSONResponse({"ok": True})
    bot_row = rows[0]

    bot_token = bot_row.get("bot_api_token")
    if not bot_token:
        return JSONResponse({"ok": True})
    owner_id = bot_row.get("owner_telegram_id")
    system_prompt = bot_row.get("system_prompt") or "Ты — полезный ассистент поддержки клиентов."

    # Подписка: истёкшая платная -> бот не отвечает.
    subscription_active = True
    expires = parse_dt(bot_row.get("subscription_expires_at"))
    if expires is not None:
        subscription_active = expires > datetime.now(timezone.utc)

    plan_id = bot_row.get("plan_id")
    free_tokens_used = bot_row.get("free_tokens_used") or 0
    is_free_tier = not plan_id

    # Автор сообщения: владелец (написал клиенту вручную) или клиент.
    sender_id = (biz_message.get("from") or {}).get("id")
    is_from_owner = bool(owner_id) and sender_id == owner_id

    conv = supabase.table("conversations").upsert(
        {
            "bot_id": bot_id,
            "customer_chat_id": chat_id,
            "customer_username": username,
            "business_connection_id": business_connection_id,
        },
        on_conflict="bot_id,customer_chat_id",
    ).execute().data[0]

    if is_from_owner:
        supabase.table("messages").insert({
            "conversation_id": conv["id"], "role": "owner", "content": display_text,
        }).execute()
        return JSONResponse({"ok": True})

    supabase.table("messages").insert({
        "conversation_id": conv["id"], "role": "customer", "content": display_text,
    }).execute()

    if conv.get("status") == "human_takeover":
        if not takeover_expired(supabase, conv["id"]):
            return JSONResponse({"ok": True})
        set_status(supabase, conv["id"], "active")
        conv["status"] = "active"

    # Не спамим владельца уведомлениями на каждое сообщение уже эскалированного диалога.
    already_awaiting = conv.get("status") == "awaiting_human"

    if not subscription_active:
        set_status(supabase, conv["id"], "awaiting_human")
        if not already_awaiting:
            await notify_owner(owner_id, username, display_text, conv["id"], HEADER_EXPIRED)
        return JSONResponse({"ok": True})

    if is_free_tier and free_tokens_used >= config.FREE_TIER_TOKEN_LIMIT:
        set_status(supabase, conv["id"], "awaiting_human")
        if not already_awaiting:
            await notify_owner_limit_reached(owner_id, config.FREE_TIER_TOKEN_LIMIT)
        return JSONResponse({"ok": True})

    used, period_start = get_usage(bot_row)
    if plan_id:
        plan_limit = get_plan_messages_limit(supabase, plan_id)
        if plan_limit is not None and used >= plan_limit:
            set_status(supabase, conv["id"], "awaiting_human")
            if not already_awaiting:
                await notify_owner(owner_id, username, display_text, conv["id"], HEADER_LIMIT)
            return JSONResponse({"ok": True})

    # Не-текст: ИИ ничего не найдёт — сразу владельцу.
    if not text:
        set_status(supabase, conv["id"], "awaiting_human")
        await notify_owner(owner_id, username, display_text, conv["id"])
        return JSONResponse({"ok": True})

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

    # Блокирующие вызовы (эмбеддинг, LLM) — в пул потоков, чтобы не стопорить event loop.
    try:
        context, _similarity = await run_in_threadpool(rag.search_knowledge_base, bot_id, text)
        reply, tokens_used = await run_in_threadpool(
            qwen_client.generate_reply, system_prompt, context, text, is_first_message
        )
    except Exception as exc:
        print(f"[business_webhook] ошибка ИИ для бота {bot_id}: {exc}")
        set_status(supabase, conv["id"], "awaiting_human")
        if not already_awaiting:
            await notify_owner(owner_id, username, text, conv["id"], HEADER_ERROR)
        return JSONResponse({"ok": True})

    if is_free_tier and tokens_used:
        try:
            supabase.table("bots").update(
                {"free_tokens_used": free_tokens_used + tokens_used}
            ).eq("id", bot_id).execute()
        except Exception as exc:
            print(f"[business_webhook] не удалось обновить free_tokens_used: {exc}")

    if is_escalation(reply):
        set_status(supabase, conv["id"], "awaiting_human")
        if not already_awaiting:
            await notify_owner(owner_id, username, text, conv["id"])
        return JSONResponse({"ok": True})

    try:
        await send_business_message(bot_token, business_connection_id, chat_id, reply)
    except Exception as exc:
        # Не отдаём 500, иначе Telegram будет повторять апдейт и ИИ ответит дважды.
        print(f"[business_webhook] не удалось отправить ответ клиенту: {exc}")
        return JSONResponse({"ok": True})

    supabase.table("messages").insert({
        "conversation_id": conv["id"], "role": "assistant", "content": reply,
    }).execute()

    # Учитываем отправленный ответ ИИ в лимите тарифа.
    try:
        supabase.table("bots").update({
            "ai_replies_used": used + 1,
            "ai_replies_period_start": period_start.isoformat(),
        }).eq("id", bot_id).execute()
    except Exception as exc:
        print(f"[business_webhook] не удалось обновить счётчик ответов: {exc}")

    return JSONResponse({"ok": True})


@app.get("/health")
async def health():
    return {"ok": True}
