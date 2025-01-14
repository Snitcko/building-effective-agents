import os
import getpass
from typing import List, Dict, Optional
import json
import requests
from pydantic import BaseModel, Field

from config import OPENAI_API_KEY, PINECONE_API_KEY, INDEX_NAME

# Helper class for running dialogue tests
class DialogueTest:
    """Simple framework for testing LLM dialogue capabilities."""
    
    def __init__(self, agent):
        """Initialize with an LLM agent to test."""
        self.agent = agent
    
    def chat(self, message: str) -> str:
        """Send a message and get the response."""
        print(f"\nUser: {message}")
        response = self.agent.process_message(message)
        print(f"Assistant: {response}")
        return response
    
class BasicLLM:
    """Basic LLM with just chat capability."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model
        
    def process_message(self, message: str) -> str:
        """Process a single message."""
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful AI assistant."
                        },
                        {
                            "role": "user",
                            "content": message
                        }
                    ]
                }
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error: {str(e)}"

class Message(BaseModel):
    """Single message in conversation history."""
    role: str
    content: str

class MemoryLLM(BasicLLM):
    """LLM with conversation memory."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        super().__init__(api_key, model)
        self.history: List[Message] = []
        
    def process_message(self, message: str) -> str:
        """Process message with conversation history."""
        try:
            # Prepare messages including history
            messages = [{"role": "system", "content": "You are a helpful marketing assistant. Answer just if you sure about correct info 100%. If not say you not sure."}]
            messages.extend([msg.dict() for msg in self.history])
            messages.append({"role": "user", "content": message})
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages
                }
            )
            response.raise_for_status()
            
            # Get response and update history
            assistant_message = response.json()['choices'][0]['message']['content']
            self.history.append(Message(role="user", content=message))
            self.history.append(Message(role="assistant", content=assistant_message))
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"


class DocumentProcessor:
    """Processes documents into chunks."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_chunks(self, text: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks."""
        if not text:
            raise ValueError("Empty text provided")
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("chunk_size must be greater than chunk_overlap")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            chunks.append({
                'text': chunk,
                'metadata': {'start_char': start, 'end_char': end}
            })
            
            start += self.chunk_size - self.chunk_overlap
            
        print(f"\nCreated {len(chunks)} chunks")
        print(f"Sample chunk (first 50 chars): {chunks[0]['text'][:50]}...")
        return chunks

class EmbeddingGenerator:
    """Handles text embedding generation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/embeddings"
        
    def create_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if not text.strip():
            raise ValueError("Empty text provided")
            
        response = requests.post(
            self.api_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": text
            }
        )
        response.raise_for_status()
        embedding = response.json()['data'][0]['embedding']
        print(f"Generated embedding, sample of first 3 dimensions: {embedding[:3]}")
        return embedding

class VectorStore:
    """Manages vector storage operations."""

    def __init__(self, api_key: str, index_name: str):
        self.api_key = api_key
        self.index_name = index_name
        self.headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": "2024-07"
        }
        self.index_host_cache = {}
        
    def describe_index(self) -> str:
        """Get or retrieve cached index host."""
        if self.index_name in self.index_host_cache:
            return self.index_host_cache[self.index_name]
            
        url = f"https://api.pinecone.io/indexes/{self.index_name}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        host = response.json()["host"]
        self.index_host_cache[self.index_name] = host
        print(f"\nConnected to Pinecone host: {host}")
        return host
        
    def query_vectors(self, 
                      query_vector: List[float], 
                      top_k: int = 3, 
                      namespace: str = "") -> List[Dict]:
        """Query for most similar vectors."""
        host = self.describe_index()
        url = f"https://{host}/query"
        data = {
            "vector": query_vector,
            "topK": top_k,
            "namespace": namespace,
            "includeMetadata": True
        }
        
        print(f"\nQuerying Pinecone for top {top_k} matches")
        response = requests.post(url, headers=self.headers, json=data)
        response.raise_for_status()
        
        matches = response.json()['matches']
        return matches
        
    def store_vectors(self, vectors: List[Dict], namespace: str = "") -> Dict:
        """Store vectors in Pinecone."""
        host = self.describe_index()
        url = f"https://{host}/vectors/upsert"
        data = {
            "vectors": vectors,
            "namespace": namespace
        }
        
        print(f"\nUploading {len(vectors)} vectors to Pinecone")
        print(f"Sample vector metadata: {vectors[0]['metadata']}")
        
        response = requests.post(url, headers=self.headers, json=data)    
        response.raise_for_status()
        result = response.json()
        print(f"Pinecone response: {result}")
        return result

def process_and_store_document(text: str, 
                               document_id: str, 
                               openai_key: str, 
                               pinecone_key: str, 
                               index_name: str) -> bool:
    """Process document and store in vector database."""
    try:
        # Initialize components
        processor = DocumentProcessor()
        embedding_gen = EmbeddingGenerator(openai_key)
        vector_store = VectorStore(pinecone_key, index_name)
        
        # Create chunks
        chunks = processor.create_chunks(text)
        
        # Generate embeddings and prepare vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = embedding_gen.create_embedding(chunk['text'])
            if embedding:
                vectors.append({
                    'id': f"{document_id}-{i}",
                    'values': embedding,
                    'metadata': {
                        'text': chunk['text'],
                        'document_id': document_id,
                        **chunk['metadata']
                    }
                })
        
        # Store vectors
        if vectors:
            vector_store.store_vectors(vectors)
            return True
        return False
            
    except Exception as e:
        print(f"Error processing document: {e}")
        return False

class RagLLM(MemoryLLM):
    """LLM with conversation memory and RAG capability."""
    
    def __init__(self, 
                 openai_key: str, 
                 pinecone_key: str, 
                 index_name: str, 
                 model: str = "gpt-4o-mini"):
        super().__init__(openai_key, model)
        self.pinecone_key = pinecone_key
        self.index_name = index_name
        self.embedding_generator = EmbeddingGenerator(openai_key)
        self.vector_store = VectorStore(pinecone_key, index_name)
        
    def _get_relevant_chunks(self, text: str, top_k: int = 3) -> List[str]:
        """Get relevant text chunks using vector search."""
        try:
            # Get query embedding
            query_vector = self.embedding_generator.create_embedding(text)
            
            # Query vector store
            matches = self.vector_store.query_vectors(
                query_vector=query_vector,
                top_k=top_k
            )
            
            # Extract texts from metadata
            return [match['metadata'].get('text', '') for match in matches]
            
        except Exception as e:
            print(f"Error getting relevant chunks: {e}")
            return []
            
    def process_message(self, message: str) -> str:
        """Process message with conversation history and relevant context."""
        try:
            # Get relevant context
            context_chunks = self._get_relevant_chunks(message)
            context = "\n".join(context_chunks)
            print(f"\nFound {len(context_chunks)} relevant chunks")
            if context_chunks:
                print(f"Sample context (first 50 chars): {context_chunks[0][:50]}...")
            
            # Prepare messages
            messages = [{
                "role": "system",
                "content": f"""You are a helpful marketing assistant. Answer just if you sure about correct info 100%. If not say you not sure. 
                Use this context when relevant:
                {context}
                """
            }]
            messages.extend([msg.model_dump() for msg in self.history])
            messages.append({"role": "user", "content": message})
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages
                }
            )
            response.raise_for_status()
            
            # Get response and update history
            assistant_message = response.json()['choices'][0]['message']['content']
            self.history.append(Message(role="user", content=message))
            self.history.append(Message(role="assistant", content=assistant_message))
            
            return assistant_message
            
        except Exception as e:
            return f"Error: {str(e)}"


class MarketingTools:
    """Provides marketing analytics calculations."""

    def get_tool_definitions(self) -> List[Dict]:
        tools = [{
            "type": "function",
            "function": {
                "name": "calculate_romi",
                "description": "Calculate Return on Marketing Investment (ROMI)",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "revenue": {
                            "type": "number",
                            "description": "Total revenue generated from marketing campaign"
                        },
                        "marketing_cost": {
                            "type": "number",
                            "description": "Total cost of marketing campaign"
                        },
                        "margin_percent": {
                            "type": "number",
                            "description": "Profit margin percentage (0-100)"
                        }
                    },
                    "required": ["revenue", "marketing_cost", "margin_percent"],
                    "additionalProperties": False
                }
            }
        }]
        return tools
    
    @staticmethod
    def calculate_romi(revenue: float, 
                       marketing_cost: float, 
                       margin_percent: float) -> float:
        """
        Calculate ROMI using the formula: ((Revenue * Margin%) - Marketing Cost) / Marketing Cost * 100
        Returns percentage value
        """
        if marketing_cost <= 0:
            raise ValueError("Marketing cost must be greater than zero")
        if not (0 <= margin_percent <= 100):
            raise ValueError("Margin percentage must be between 0 and 100")
            
        margin_multiplier = margin_percent / 100
        profit = revenue * margin_multiplier - marketing_cost
        romi = (profit / marketing_cost) * 100
        return round(romi, 2)

class ToolEnabledLLM(RagLLM):
    """LLM with memory, RAG and tool usage capability."""
    
    def __init__(self, 
                 openai_key: str, 
                 pinecone_key: str, 
                 index_name: str, 
                 model: str = "gpt-4o-mini"):
        super().__init__(openai_key, pinecone_key, index_name, model)
        self.tools = MarketingTools()
    
    def process_message(self, message: str) -> str:
        """Process message with tools, memory and RAG."""
        try:
            # Get relevant context
            context_chunks = self._get_relevant_chunks(message)
            context = "\n".join(context_chunks)
            print(f"\nFound {len(context_chunks)} relevant chunks")
            if context_chunks:
                print(f"Sample context (first 100 chars): {context_chunks[0][:100]}...")
            
            # Prepare messages
            messages = [{
                "role": "system",
                "content": f"""You are a marketing analytics assistant. 
                Use this context when relevant:
                {context}
                
                When asked about marketing metrics, use the calculate_romi tool.
                ROMI (Return on Marketing Investment) shows the profitability of marketing spending.
                """
            }]
            messages.extend([msg.dict() for msg in self.history])
            messages.append({"role": "user", "content": message})
            
            # Make API request with tools
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools.get_tool_definitions()
                }
            )
            response.raise_for_status()
            
            # Process response and tool calls
            response_data = response.json()
            assistant_message = response_data['choices'][0]['message']
            
            if tool_calls := assistant_message.get('tool_calls'):
                print("\nProcessing tool calls...")
                tool_results = self._handle_tool_calls(tool_calls)
                
                # Add tool results to conversation
                messages.extend([
                    {
                        "role": "assistant",
                        "content": assistant_message.get('content'),
                        "tool_calls": tool_calls
                    },
                    {
                        "role": "tool",
                        "content": json.dumps(tool_results),
                        "tool_call_id": tool_calls[0]['id']
                    }
                ])
                
                # Get final response with tool results
                final_response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages
                    }
                )
                final_response.raise_for_status()
                assistant_message = final_response.json()['choices'][0]['message']
            
            # Update conversation history
            self.history.append(Message(role="user", content=message))
            self.history.append(Message(role="assistant", content=assistant_message['content']))
            
            return assistant_message['content']
            
        except Exception as e:
            return f"Error: {str(e)}"
            
    def _handle_tool_calls(self, tool_calls: List[Dict]) -> Dict:
        """Process tool calls and return results."""
        results = {}
        
        for call in tool_calls:
            if call['function']['name'] == 'calculate_romi':
                try:
                    args = json.loads(call['function']['arguments'])
                    result = self.tools.calculate_romi(
                        revenue=args['revenue'],
                        marketing_cost=args['marketing_cost'],
                        margin_percent=args['margin_percent']
                    )
                    print(f"Calculated ROMI: {result}%")
                    results[call['id']] = result
                except Exception as e:
                    results[call['id']] = f"Error calculating ROMI: {str(e)}"
                    
        return results