from supabase import create_client
import config
import gemini_client

supabase = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)


def search_knowledge_base(bot_id: str, question: str, match_count: int = 3):
    """
    Векторный поиск по базе знаний конкретного бота через Supabase RPC-функцию
    `match_knowledge_base` (см. schema.sql), использующую pgvector <-> оператор.

    Возвращает (context_text, best_similarity) — конкатенированный контекст
    и максимальную схожесть среди найденных фрагментов.
    """
    embedding = gemini_client.embed_text(question)

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
