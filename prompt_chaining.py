from typing import List, Dict, Optional
from pydantic import BaseModel, HttpUrl
import json
import requests

# Data Models
class ArticleContent(BaseModel):
    """Represents scraped and processed article content"""
    url: HttpUrl
    title: str
    content: str
    metadata: Dict

class ContentBrief(BaseModel):
    """Represents the content brief/specification"""
    main_topics: List[str]
    target_audience: str
    key_messages: List[str]
    tone_guidelines: str
    hashtags: List[str]
    constraints: Dict[str, str]  # platform-specific constraints
    source_article: ArticleContent

class PostDraft(BaseModel):
    """Represents the initial post draft"""
    content: str
    hashtags: List[str]
    brief: ContentBrief
    metadata: Dict

class FinalPost(BaseModel):
    """Represents the final optimized post"""
    content: Dict[str, str]  # language -> content mapping
    hashtags: List[str]
    engagement_metrics: Dict[str, float]  # predicted engagement metrics
    platform_specific: Dict[str, Dict]  # platform-specific formatting


class WebScraper:
    """Handles article scraping using Jina Reader API"""
    
    def __init__(self, jina_api_key: str):
        self.api_key = jina_api_key
        self.api_url = "https://r.jina.ai"

    async def process_response(self, response) -> dict:
        """Process API response and return structured data"""
        try:
            if response.status_code == 200:
                return {
                    'text': response.text,
                    'status': 'success'
                }
            else:
                print(f"Error status: {response.status_code}")
                print(f"Response headers: {response.headers}")
                print(f"Response content: {response.text[:500]}")
                response.raise_for_status()
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            raise
        
    async def scrape(self, url: str) -> ArticleContent:
        """Scrape article content from URL using Jina Reader API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Return-Format": "text"
            }
            
            data = {
                "url": url
            }
            
            print(f"Making request to Jina AI...")
            print(f"URL to scrape: {url}")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data 
            )
            
            result = await self.process_response(response)
            
            return ArticleContent(
                url=url,
                title=url.split('/')[-1] or url,
                content=result['text'],
                metadata={
                    "source": "jina_reader",
                    "scrape_time": "now",
                    "status": result['status']
                }
            )
            
        except Exception as e:
            print(f"Full error details: {str(e)}")
            raise ValueError(f"Failed to scrape URL {url}: {str(e)}")

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
        print(f"store_vectors")
        if self.index_name in self.index_host_cache:
            return self.index_host_cache[self.index_name]
            
        url = f"https://api.pinecone.io/indexes/{self.index_name}"
        print(f"Describing Pinecone index: {self.index_name}")
        print(f"URL: {url}")
        print(f"response = requests.get(url, headers=self.headers)")
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Response Content: {response.text}")
        response.raise_for_status()
        
        host = response.json()["host"]
        print(f"host = {host}")
        self.index_host_cache[self.index_name] = host
        print(f"\nConnected to Pinecone host: {host}")
        return host
        
    def query_vectors(self, 
                      query_vector: List[float], 
                      top_k: int = 2, 
                      namespace: str = "") -> List[Dict]:
        """Query for most similar vectors."""
        host = self.describe_index()
        print(f"query_vectors")
        print(f"Namespace: {namespace}")
        print(f"host: {host}")
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
        print(f"store_vectors")
        host = self.describe_index()
        print(f"Namespace: {namespace}")
        print(f"host: {host}")
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

class RagProcessor:
    """Handles RAG operations"""
    
    def __init__(self, openai_key: str, pinecone_key: str, index_name: str):
        self.processor = DocumentProcessor()
        self.embedding_gen = EmbeddingGenerator(openai_key)
        self.vector_store = VectorStore(pinecone_key, index_name)
    
    async def store_article(self, article: ArticleContent) -> str:
        """Process article and store in vector database"""
        try:
            # Create chunks
            chunks = self.processor.create_chunks(article.content)
            
            # Generate embeddings and prepare vectors
            vectors = []
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_gen.create_embedding(chunk['text'])
                if embedding:
                    vector_id = f"article_{hash(article.url)}_{i}"
                    vectors.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': {
                            'text': chunk['text'],
                            'url': str(article.url),
                            'title': article.title,
                            **chunk['metadata']
                        }
                    })
            
            # Store vectors
            if vectors:
                self.vector_store.store_vectors(vectors)
                return vectors[0]['id']  # Return first chunk ID
            return ""
            
        except Exception as e:
            raise ValueError(f"Failed to store article: {str(e)}")
    
    async def get_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Get relevant context for query"""
        try:
            query_embedding = self.embedding_gen.create_embedding(query)
            return self.vector_store.query_vectors(query_embedding, top_k)
        except Exception as e:
            raise ValueError(f"Failed to get context: {str(e)}")


class ContentAnalyst:
    """Creates content brief from article"""
    
    def __init__(self, api_key: str, rag_processor=None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.rag_processor = rag_processor
        
    async def process(self, article: ArticleContent) -> ContentBrief:
        try:
            # Get relevant context if RAG is available
            context = ""
            if self.rag_processor:
                matches = await self.rag_processor.get_context(article.content)
                context = "\n\n".join(m["metadata"].get("text", "") for m in matches if "metadata" in m)
            
            # Make API request
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
                            "content": """Create a social media brief in this exact JSON format:
                            {
                                "main_topics": ["topic1", "topic2"],
                                "target_audience": "description",
                                "key_messages": ["msg1", "msg2"],
                                "tone_guidelines": "description",
                                "hashtags": ["#tag1", "#tag2"],
                                "constraints": {"platform": "rules"}
                            }"""
                        },
                        {
                            "role": "user",
                            "content": f"Article: {article.content[:2000]}"
                        }
                    ]
                }
            )
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"Raw content from API: {content}") 
            
            result = json.loads(content)
            
            return ContentBrief(
                main_topics=result['main_topics'],
                target_audience=result['target_audience'],
                key_messages=result['key_messages'],
                tone_guidelines=result['tone_guidelines'],
                hashtags=result['hashtags'],
                constraints=result['constraints'],
                source_article=article
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
        
class ContentWriter:
    """Creates post draft from brief"""
    
    def __init__(self, api_key: str, rag_processor=None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.rag_processor = rag_processor
        
    async def process(self, brief: ContentBrief) -> PostDraft:
        try:
            # Get additional context if RAG is available
            context = ""
            if self.rag_processor:
                matches = await self.rag_processor.get_context(brief.source_article.content)
                context = "\n\n".join(m["metadata"].get("text", "") for m in matches if "metadata" in m)
            
            # Make API request
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
                            "content": "You are a social media copywriter. Create engaging content based on the brief."
                        },
                        {
                            "role": "user",
                            "content": f"""
                            Topics: {brief.main_topics}
                            Audience: {brief.target_audience}
                            Messages: {brief.key_messages}
                            Tone: {brief.tone_guidelines}
                            
                            Additional context: {context}
                            
                            Create an engaging social media post."""
                        }
                    ]
                }
            )
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            return PostDraft(
                content=content,
                hashtags=brief.hashtags,
                brief=brief,
                metadata={"timestamp": "now"}
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

class ContentOptimizer:
    """Optimizes and translates post"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        
    async def process(self, draft: PostDraft) -> FinalPost:
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
                            "content": """Optimize and translate this post. Return JSON in format:
                            {
                                "translations": {
                                    "en": "english text",
                                    "es": "spanish text"
                                },
                                "engagement_metrics": {
                                    "expected_likes": 100,
                                    "expected_shares": 50
                                },
                                "platform_versions": {
                                    "twitter": {"text": "twitter text", "type": "tweet"},
                                    "linkedin": {"text": "linkedin text", "type": "post"}
                                }
                            }"""
                        },
                        {
                            "role": "user",
                            "content": f"""Post: {draft.content}
                            Tone: {draft.brief.tone_guidelines}
                            Audience: {draft.brief.target_audience}"""
                        }
                    ]
                }
            )
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            print(f"Raw content from API: {content}")
            result = json.loads(content)
            
            return FinalPost(
                content=result['translations'],
                hashtags=draft.hashtags,
                engagement_metrics=result['engagement_metrics'],
                platform_specific=result['platform_versions']
            )
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise

class ContentPipeline:
    """Orchestrates the content processing pipeline"""
    
    def __init__(self, openai_key: str, jina_key: str, pinecone_key: str, index_name: str):
        # Initialize components
        self.scraper = WebScraper(jina_key)
        self.rag = RagProcessor(openai_key, pinecone_key, index_name)
        
        # Initialize agents with RAG
        self.analyst = ContentAnalyst(openai_key, self.rag)
        self.writer = ContentWriter(openai_key, self.rag)
        self.optimizer = ContentOptimizer(openai_key)
    
    async def process_url(self, url: str) -> FinalPost:
        """Process URL through the pipeline"""
        try:
            # 1. Scrape article
            print("1. Scraping article...")
            article = await self.scraper.scrape(url)
            print("✓ Article scraped")
            
            # 2. Store in RAG system
            print("2. Storing in RAG system...")
            await self.rag.store_article(article)
            print("✓ Article stored in RAG")
            
            # 3. Generate brief
            print("3. Generating content brief...")
            brief = await self.analyst.process(article)
            print("✓ Content brief created")
            
            # 4. Create draft
            print("4. Creating post draft...")
            draft = await self.writer.process(brief)
            print("✓ Post draft created")
            
            # 5. Optimize and finalize
            print("5. Optimizing content...")
            final = await self.optimizer.process(draft)
            print("✓ Content optimized")
            
            return final
            
        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
