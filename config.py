import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# --- Qwen (Alibaba Cloud DashScope, OpenAI-совместимый режим) ---
QWEN_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

TELEGRAM_SERVICE_BOT_TOKEN = os.getenv("TELEGRAM_SERVICE_BOT_TOKEN", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

# Секрет вебхука Telegram (secret_token). ОБЯЗАТЕЛЕН: без него сервис
# отклоняет все запросы, чтобы никто не мог слать поддельные апдейты.
# Должен совпадать с TELEGRAM_WEBHOOK_SECRET на стороне Next.js.
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.75"))

# --- Лимит бесплатного тарифа (токены ответов ИИ) ---
# Должен совпадать с FREE_TIER_TOKEN_LIMIT на стороне Next.js.
FREE_TIER_TOKEN_LIMIT = int(os.getenv("FREE_TIER_TOKEN_LIMIT", "15000"))

# Запасные лимиты ответов ИИ за 30 дней, если тарифа нет в таблице `plans`.
# Тарифа нет в словаре = без лимита.
DEFAULT_PLAN_MESSAGE_LIMITS = {"start": 50, "business": 5000}

# Через сколько часов после последнего ответа владельца ИИ снова начинает
# отвечать клиенту (диалог в статусе human_takeover). 0 = не возвращать.
HUMAN_TAKEOVER_HOURS = float(os.getenv("HUMAN_TAKEOVER_HOURS", "12"))
