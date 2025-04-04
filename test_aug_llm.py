from typing import List
from augmented_llm import (
    ToolEnabledLLM, 
    process_and_store_document
)
from config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME

def prepare_marketing_context():
    """Загрузка контекста в векторное хранилище"""
    marketing_text = """
    Marketing metrics are essential for evaluating the effectiveness of marketing strategies and campaigns. Here are some key metrics:
    **Return on Marketing Investment (ROMI)**: 
    Measures the revenue generated for every dollar spent on marketing. 
    Formula: (Revenue - Marketing Cost) / Marketing Cost.
    """
    
    return process_and_store_document(
        text=marketing_text, 
        document_id="marketing_metrics_context",
        openai_key=OPENAI_API_KEY, 
        pinecone_key=PINECONE_API_KEY, 
        index_name=INDEX_NAME
    )

def test_tool_enabled_llm():
    # prepare context
    context_loaded = prepare_marketing_context()
    print(f"Context is loaded: {'✓' if context_loaded else '✗'}")

    llm = ToolEnabledLLM(
        openai_key=OPENAI_API_KEY, 
        pinecone_key=PINECONE_API_KEY, 
        index_name=INDEX_NAME
    )

    # Test messages
    test_messages = [
        "What is ROMI?",
        "Calculate ROMI for a campaign with revenue of 50,000, marketing expenses of 10,000 and a margin of 25%"
    ]

    for message in test_messages:
        response = llm.process_message(message)
        print(f"\n🔹 Request: {message}")
        print(f"🔸 Response: {response}")

if __name__ == "__main__":
    test_tool_enabled_llm()