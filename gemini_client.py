import google.generativeai as genai
import config

genai.configure(api_key=config.GEMINI_API_KEY)


def generate_reply(system_prompt: str, context: str, question: str) -> str:
    """Генерирует ответ клиенту на основе найденного контекста базы знаний."""
    model = genai.GenerativeModel(
        model_name=config.GEMINI_CHAT_MODEL,
        system_instruction=system_prompt,
    )
    prompt = (
        f"Контекст из базы знаний:\n{context}\n\n"
        f"Вопрос клиента: {question}\n\n"
        "Ответь кратко и по существу, опираясь только на контекст выше. "
        "Если в контексте нет ответа — не придумывай."
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def embed_text(text: str) -> list[float]:
    result = genai.embed_content(model=f"models/{config.GEMINI_EMBEDDING_MODEL}", content=text)
    return result["embedding"]
