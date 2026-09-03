from supabase import create_client
import config
import qwen_client

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


def search_knowledge_base(bot_id: str, question: str, match_count: int = 3):
    """
    Векторный поиск по базе знаний конкретного бота через Supabase RPC-функцию
    `match_knowledge_base` (см. schema.sql), использующую pgvector <-> оператор.
    Эмбеддинги теперь генерируются через Qwen (text-embedding-v3, 1024 измерения) —
    см. lib/supabase/schema.sql, столбец embedding обновлён до vector(1024).

    Возвращает (context_text, best_similarity) — конкатенированный контекст
    и максимальную схожесть среди найденных фрагментов.
    """
    # ВАЖНО: Qwen embeddings API падает с 400 "input.texts should not be null",
    # если передать пустую строку (например, клиент прислал стикер/фото/голосовое
    # без текста — biz_message.get("text", "") тогда возвращает ""). Раньше это
    # роняло весь webhook с 500-й ошибкой. Теперь просто считаем, что контекст
    # не найден, не дергая Qwen вообще.
    question = (question or "").strip()
    if not question:
        return "", 0.0

    supabase = get_supabase()
    embedding = qwen_client.embed_text(question)

    response = supabase.rpc(
        "match_knowledge_base",
        {
            "query_embedding": embedding,
            "match_bot_id": bot_id,
            "match_count": match_count,
        },
    ).execute()

    rows = response.data or []
    if not rows:
        return "", 0.0

    context = "\n---\n".join(row["content"] for row in rows)
    best_similarity = max(row["similarity"] for row in rows)
    return context, best_similarity
