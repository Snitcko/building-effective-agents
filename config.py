import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

# Проверяем наличие необходимых переменных
def get_env_variable(var_name: str) -> str:
    """Get environment variable or raise exception."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value

# Экспортируем переменные
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY")
INDEX_NAME = get_env_variable("INDEX_NAME")