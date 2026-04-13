import uuid
import asyncio
from VDBWrapper.dedupe_manager import DedupeManager
from SQL_UTILS.SQLITE_WRAPPER import AsyncVecSQLite as SQLiteWrapper
from OpenRouterWrapper.ChatBot import OpenRouterChat
import os

"""
we need a workspace construct the knowledgebase whole process
- We need chunk manager to process the documents and assign each chunk an id
- We need entity manager to extract entities from each chunk and assign each entity an id, type, description
"""
class AsyncNormalizeEntitiesWorkspace:
    def __init__(self, db_name, DIM=1024, model="meta-llama/llama-3.1-8b-instruct", embedding_model="baai/bge-large-en-v1.5", MAX_TRIES=3, verbose=False):
        self.db_name = db_name
        self.max_tries = MAX_TRIES
        self.DIM = DIM
        self.semaphore = asyncio.Semaphore(10) 
        self.tokenizer_model = model
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.llm = OpenRouterChat(api_key=os.getenv("OPENROUTER_API_KEY"), model=model)



    

   
    async def __aenter__(self):
        if self.verbose:
            print("Constructing the ShadowRAG database!")

        async with SQLiteWrapper(db_name=self.db_name) as self.DB:
            cluster_json_list, entity_ids = await self.DB.cluster_similar_entities()
            print(entity_ids)
            DEDUPE_MANAGER = DedupeManager(clusters=cluster_json_list,
                                            entity_ids=entity_ids,
                                              DB=self.DB, 
                                               max_tries=self.max_tries, 
                                               semaphore=self.semaphore,
                                               llm_func=self.llm.generate)

            await DEDUPE_MANAGER.batch_merge_choices()
            # Then, we add in the singleton shadow entity ids
            if len(DEDUPE_MANAGER.entity_ids) > 0:
                await DEDUPE_MANAGER.batch_singleton_shadows()

            await DEDUPE_MANAGER.batch_entity_swapping()
            

            

        return self


    async def __aexit__(self, exc_type, exc, tb):
        await self.llm.close()
        if self.verbose:
            print("Finished constructing the ShadowRAG database!")
    

  