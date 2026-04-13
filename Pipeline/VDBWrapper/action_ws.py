import uuid
import asyncio
from VDBWrapper.action_manager import ActionManager
from SQL_UTILS.SQLITE_WRAPPER import AsyncVecSQLite as SQLiteWrapper
from OpenRouterWrapper.ChatBot import OpenRouterChat
import os

"""
we need a workspace construct the knowledgebase whole process
- We need chunk manager to process the documents and assign each chunk an id
- We need entity manager to extract entities from each chunk and assign each entity an id, type, description
"""
class AsyncActionWorkspace:
    def __init__(self, db_name, sensitive_action_dict, replacement_dict,  DIM=1024, model="meta-llama/llama-3.1-8b-instruct", embedding_model="baai/bge-large-en-v1.5", MAX_TRIES=3, verbose=False):
        self.db_name = db_name
        self.max_tries = MAX_TRIES
        self.action_dict = sensitive_action_dict
        self.action_replacements = replacement_dict
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
           ACTION_MANAGER = ActionManager(sensitive_actions=self.action_dict, replacements=self.action_replacements, DB=self.DB, max_tries=self.max_tries, semaphore=self.semaphore, llm_func=self.llm.generate)
           action_chunks = await ACTION_MANAGER.get_action_chunks()
           await ACTION_MANAGER.extract_sensitive_actions_from_chunk(action_chunks['chunk-cb1d8e5514e4922dde6ff0c87ca8f638'], "chunk-cb1d8e5514e4922dde6ff0c87ca8f638")

           
             


        return self


    async def __aexit__(self, exc_type, exc, tb):
        await self.llm.close()
        if self.verbose:
            print("Finished constructing the ShadowRAG database!")
    

  