import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# --- Qwen (Alibaba Cloud DashScope, OpenAI-совместимый режим) ---
# Полностью заменяет прежнюю интеграцию с Gemini.
QWEN_API_KEY = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", ""))
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3")

TELEGRAM_SERVICE_BOT_TOKEN = os.getenv("TELEGRAM_SERVICE_BOT_TOKEN", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.75"))
