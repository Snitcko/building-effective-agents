from typing import List
from augmented_llm import (
    ToolEnabledLLM, 
    process_and_store_document
)
from config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME

def prepare_marketing_context():
    """Загрузка контекста в векторное хранилище"""
    marketing_text = """
    Маркетинговые метрики играют критическую роль в оценке эффективности кампаний. 
    ROMI (Return on Marketing Investment) показывает прибыльность маркетинговых инвестиций. 
    Он рассчитывается как отношение прибыли к маркетинговым затратам, выраженное в процентах.
    
    Например, если кампания с бюджетом 10 000$ принесла выручку 50 000$ и маржинальность 25%, 
    то ROMI будет положительным и покажет реальную эффективность инвестиций.
    """
    
    return process_and_store_document(
        text=marketing_text, 
        document_id="marketing_metrics_context",
        openai_key=OPENAI_API_KEY, 
        pinecone_key=PINECONE_API_KEY, 
        index_name=INDEX_NAME
    )

def test_tool_enabled_llm():
    # Подготовка контекста
    context_loaded = prepare_marketing_context()
    print(f"Контекст загружен: {'✓' if context_loaded else '✗'}")

    # Инициализация LLM
    llm = ToolEnabledLLM(
        openai_key=OPENAI_API_KEY, 
        pinecone_key=PINECONE_API_KEY, 
        index_name=INDEX_NAME
    )

    # Тестовый диалог
    test_messages = [
        "Что такое ROMI?",
        "Посчитай ROMI для кампании с выручкой 50000, маркетинговыми расходами 10000 и маржей 25%"
    ]

    for message in test_messages:
        response = llm.process_message(message)
        print(f"\n🔹 Запрос: {message}")
        print(f"🔸 Ответ: {response}")

if __name__ == "__main__":
    test_tool_enabled_llm()