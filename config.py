import os
from dotenv import load_dotenv

load_dotenv()

def get_env_variable(var_name: str) -> str:
    """Get environment variable or raise exception."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value

OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
PINECONE_API_KEY = get_env_variable("PINECONE_API_KEY")
INDEX_NAME = get_env_variable("INDEX_NAME")
JINA_API_KEY = get_env_variable("JINA_API_KEY")