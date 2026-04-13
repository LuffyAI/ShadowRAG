import uuid
import asyncio
from VDBWrapper.chunk_manager import ChunkManager
from VDBWrapper.entity_manager import EntityManager
from SQL_UTILS.SQLITE_WRAPPER import AsyncVecSQLite as SQLiteWrapper
from OpenRouterWrapper.ChatBot import OpenRouterChat
import os

"""
we need a workspace construct the knowledgebase whole process
- We need chunk manager to process the documents and assign each chunk an id
- We need entity manager to extract entities from each chunk and assign each entity an id, type, description
"""
class AsyncShadowRAGConstructionWorkspace:
    def __init__(self, db_name, documents, sensitive_entity_dict,  DIM=1024, model="meta-llama/llama-3.1-8b-instruct", embedding_model="baai/bge-large-en-v1.5", MAX_TRIES=3, verbose=False):
        self.db_name = db_name
        self.max_tries = MAX_TRIES
        self.entity_dict = sensitive_entity_dict
        self.documents = documents
        self.DIM = DIM
        self.semaphore = asyncio.Semaphore(10) 
        self.tokenizer_model = model
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.llm = OpenRouterChat(api_key=os.getenv("OPENROUTER_API_KEY"), model=model)

    async def batch_insert_entities(self, entity_list):
        results = []

        async def process_batch(batch, idx):
            tries = 0

            while tries < self.max_tries:
                try:
                    async with self.semaphore:
                        await self.DB.insert_entities(batch)
                    return True

                except Exception as e:
                    tries += 1
                    print(f"[Batch {idx}] Retry {tries}: {e}")

                    if tries < self.max_tries:
                        await asyncio.sleep(20)
                    else:
                        print(f"[Batch {idx}] FAILED after {self.max_tries} tries")
                        return False

        tasks = [
            process_batch(batch, idx)
            for idx, batch in enumerate(entity_list)
        ]

        results = await asyncio.gather(*tasks)

        return results

    async def __aenter__(self):
        if self.verbose:
            print("Constructing the ShadowRAG database!")

        async with SQLiteWrapper(db_name=self.db_name) as self.DB:
           CHUNK_MANAGER = ChunkManager(document=self.documents, DB=self.DB, max_tries=self.max_tries, semaphore=self.semaphore)

           # First, we process all text chunks
           await CHUNK_MANAGER.insert_chunks_into_db()

           # Second, for each text_chunk, we must extract their entities, if any exist. 
           if CHUNK_MANAGER.chunks:
             ENTITY_MANAGER = EntityManager(chunks=CHUNK_MANAGER.chunks, semaphore=self.semaphore, max_tries=self.max_tries, sensitive_class_dict=self.entity_dict, DB=self.DB, llm_func=self.llm.generate)
             await ENTITY_MANAGER.process_entities()

             # If sensitive entities exist, we must batch insert them into the database
             if ENTITY_MANAGER.entity_list:
                await self.batch_insert_entities(ENTITY_MANAGER.entity_list)

             # If sensitive actions exist, we must batch them into the database
             


        return self


    async def __aexit__(self, exc_type, exc, tb):
        await self.llm.close()
        if self.verbose:
            print("Finished constructing the ShadowRAG database!")
    

  