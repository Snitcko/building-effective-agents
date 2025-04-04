import asyncio
from prompt_chaining import ContentPipeline 
from config import OPENAI_API_KEY, JINA_API_KEY, PINECONE_API_KEY, INDEX_NAME


async def main():
    try:
        pipeline = ContentPipeline(
            openai_key=OPENAI_API_KEY,
            jina_key=JINA_API_KEY,
            pinecone_key=PINECONE_API_KEY,
            index_name=INDEX_NAME
        )

        url = "https://www.anthropic.com/research/building-effective-agents"
        
        final_post = await pipeline.process_url(url)

        print("\nFinal post:")
        for lang, content in final_post.content.items():
            print(f"\nLanguage {lang}:")
            print(content[:300]) 
        
        print("\nHashtags:", final_post.hashtags)
        print("\nFinal:", final_post.engagement_metrics)

    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    asyncio.run(main())
    
    
# # Test full pipeline
# async def test_full_pipeline():
#     print("\n🚀 Testing Full Pipeline")
    
#     # Initialize pipeline
#     pipeline = ContentPipeline(
#         openai_key=OPENAI_API_KEY,
#         jina_key=JINA_API_KEY,
#         pinecone_key=PINECONE_API_KEY,
#         index_name=INDEX_NAME
#     )
    
#     # Test URL
#     url = "https://www.anthropic.com/research/building-effective-agents"
    
#     logging.info(f"Processing URL: {url}")
    
#     try:
#         print("\n1️⃣ Processing URL...")
#         final_post = await pipeline.process_url(url)
        
#         print("\n📊 Results:")
        
#         print("\n🎯 Available Languages:")
#         for lang, content in final_post.content.items():
#             print(f"\n{lang.upper()}:")
#             print(f"Length: {len(content)} characters")
#             print("Preview:")
#             print(content[:200] + "...")
        
#         print("\n#️⃣ Hashtags:")
#         print(", ".join(final_post.hashtags))
        
#         print("\n💡 Platform Versions:")
#         for platform, content in final_post.platform_specific.items():
#             print(f"\n{platform.upper()}:")
#             print(f"Preview: {content['text'][:100]}...")
        
#         print("\n📈 Engagement Metrics:")
#         for metric, value in final_post.engagement_metrics.items():
#             print(f"- {metric}: {value}")
        
#         return final_post
        
#     except requests.RequestException as e:
#         logging.error(f"Network error: {e}")
#         raise
#     except json.JSONDecodeError as e:
#         logging.error(f"JSON parsing error: {e}")
#         raise

# # Run the test
#result = test_full_pipeline()
