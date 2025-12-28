import asyncio
from time import sleep
from pymongo import AsyncMongoClient

MONGODB_URI = 'mongodb://mongo2:27017/?replicaSet=mongo2'

async def create_search_index():

    atlas_client = AsyncMongoClient(MONGODB_URI)

    await atlas_client.admin.command("ping")

    collection = atlas_client["rag_db"]["chunks"]

    # 1. Vector Search Index
    vector_index = {
        "name": "vector_index",
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 1024,  # Qwen3-Embedding-0.6B
                    "similarity": "cosine"
                }
            ]
        }
    }

    # 2. Atlas Search (Text) Index
    text_index = {
        "name": "text_index",
        "definition": {
            "mappings": {
                "dynamic": False,
                "fields": {
                    "content": {
                        "type": "string",
                        "analyzer": "lucene.standard"
                    }
                }
            }
        }
    }

    vector_result = await collection.create_search_index(vector_index)
    print(f"✅ Vector index created: {vector_result}")

    text_result = await collection.create_search_index(text_index)
    print(f"✅ Text index created: {text_result}")


async def check_index():

    atlas_client = AsyncMongoClient(MONGODB_URI)

    await atlas_client.admin.command("ping")
   
    collection = atlas_client["rag_db"]["chunks"]

    indexes_cursor = await collection.list_search_indexes()
    indexes = None
    print("\n📋 All Search Indexes:")

    flag = False
    while True:
        indexes = await indexes_cursor.to_list(None)

        statuses = set()
        for idx in indexes:
            status = idx.get('status', 'unknown')
            print(f"  - {idx['name']}: {status}")
            statuses.add(status)
        
        if 'PENDING' not in statuses:
            flag = True
        if flag:
            break
        else:
            print('waiting...')
            sleep(10)



asyncio.run(create_search_index())

asyncio.run(check_index())
