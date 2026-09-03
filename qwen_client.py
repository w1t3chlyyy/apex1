from openai import OpenAI
import config

# Qwen (Alibaba Cloud DashScope) предоставляет OpenAI-совместимый REST API,
# поэтому мы используем стандартный пакет `openai`, просто указав другой
# base_url и ключ DashScope/Qwen. Полностью заменяет google-generativeai.
_client = OpenAI(
    api_key=config.QWEN_API_KEY,
    base_url=config.QWEN_BASE_URL,
)


def generate_reply(system_prompt: str, context: str, question: str) -> str:
    """Генерирует ответ клиенту на основе найденного контекста базы знаний (Qwen)."""
    prompt = (
        f"Контекст из базы знаний:\n{context}\n\n"
        f"Вопрос клиента: {question}\n\n"
        "Ответь кратко и по существу, опираясь только на контекст выше. "
        "Если в контексте нет ответа — не придумывай."
    )
    response = _client.chat.completions.create(
        model=config.QWEN_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def embed_text(text: str) -> list[float]:
    response = _client.embeddings.create(
        model=config.QWEN_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding
