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
    """Генерирует ответ клиенту.

    РАНЬШЕ: промпт жёстко запрещал модели отвечать, если в контексте базы
    знаний не было прямого ответа ("не придумывай"). Из-за этого бот
    отвечал ТОЛЬКО на вопросы, точно совпадающие с базой знаний, а на всё
    остальное ("как дела", общие вопросы) — либо молчал, либо уходил в
    эскалацию на оператора.

    ТЕПЕРЬ: если контекст есть — отвечаем строго по нему (как раньше).
    Если контекста нет — бот всё равно отвечает, используя свою роль и
    общие знания, и только предупреждает, если вопрос требует уточнения
    индивидуальных деталей (цена, наличие и т.п.), которые он точно не
    может знать.
    """
    if context:
        prompt = (
            f"Контекст из базы знаний:\n{context}\n\n"
            f"Вопрос клиента: {question}\n\n"
            "Ответь кратко и по существу, опираясь на контекст выше. "
            "Если в контексте нет прямого ответа на вопрос, всё равно вежливо "
            "помоги клиенту, используя общие знания."
        )
    else:
        prompt = (
            f"Вопрос клиента: {question}\n\n"
            "Точной информации по этому вопросу в базе знаний компании нет. "
            "Ответь вежливо и по существу, используя свои общие знания и роль, "
            "заданную в системной инструкции. Если вопрос требует конкретных "
            "деталей бизнеса (точная цена, наличие товара, персональные данные), "
            "которые ты не можешь знать точно — честно скажи об этом и предложи "
            "уточнить у менеджера, но не отказывайся отвечать полностью."
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
