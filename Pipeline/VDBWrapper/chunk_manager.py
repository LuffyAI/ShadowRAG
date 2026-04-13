from __future__ import annotations
from typing import Any
from hashlib import md5
from OpenRouterWrapper.EmbeddingModel import OpenRouterEmbedding
import asyncio

class ChunkManager:
    def __init__(self, semaphore, document=None, DB=None, max_tries=3):
        self.content = document
        self.max_tries = max_tries
        self.semaphore = semaphore
        self.DB = DB
        self.chunks = {}
        self.shadow_chunks = {}
    
    async def chunking_by_token_size(
        self,
        overlap_token_size: int = 128,
        max_token_size: int = 1024,
    ) -> list[dict[str, Any]]:
        
        # Prevent invalid configuration
        if overlap_token_size >= max_token_size:
            raise ValueError("overlap_token_size must be smaller than max_token_size")

        # Encode full content into tokens
        tokens = self.DB.tokenizer.encode(self.content)
        results: list[dict[str, Any]] = []

        # Step size controls overlap between chunks
        step = max_token_size - overlap_token_size

        # Iterate over tokens with overlap
        for index, start in enumerate(range(0, len(tokens), step)):
            
            # Slice token window
            token_chunk = tokens[start : start + max_token_size]

            # Decode back into text
            chunk_content = self.DB.tokenizer.decode(token_chunk)

            results.append(
                {
                    "tokens": len(token_chunk),  
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
        return results
        
    async def process_documents(self, max_tokens=1024, overlap=128):
        # Generate token-based chunks from the content
        chunks = await self.chunking_by_token_size(
            max_token_size=max_tokens,
            overlap_token_size=overlap,
        )

        # Assign a unique ID to each chunk and store it
        for chunk in chunks:
            # Create a hash of the chunk content (deterministic ID)
            chunk_hash = md5(chunk["content"].encode("utf-8")).hexdigest()

            # Prefix the hash to form a readable chunk ID
            chunk_id = f"chunk-{chunk_hash}"

            exists = await self.DB.does_chunk_id_exist(chunk_id)

            if not exists:
                self.chunks[chunk_id] = chunk

    async def insert_chunks_into_db(self):
        """
        Inserts chunks into the database with semaphore + retry logic.
        """

        if self.DB is None:
            raise ValueError("Database connection (DB) is not provided")

        await self.process_documents()

        async def process_single(chunk_id, chunk):
            tries = 0
            content = chunk["content"]

            while tries < self.max_tries:
                try:
                    async with self.semaphore:
                        print(f"Inserting chunk {chunk_id} into database...")
                        await self.DB.insert_new_text_chunk(chunk_id, content)
                    return True

                except Exception as e:
                    tries += 1
                    print(f"[Chunk {chunk_id}] Retry {tries}: {e}")

                    if tries < self.max_tries:
                        await asyncio.sleep(20)
                    else:
                        print(f"[Chunk {chunk_id}] FAILED after {self.max_tries} tries")
                        return False

        tasks = [
            process_single(chunk_id, chunk)
            for chunk_id, chunk in self.chunks.items()
        ]

        await asyncio.gather(*tasks)

    # async def insert_chunks_into_db(self):
    #     """
    #     Inserts a text chunk into the database.
    #     """
    #     if self.DB is None:
    #         raise ValueError("Database connection (DB) is not provided")
        
    #     await self.process_documents()

    #     for chunk_id, chunk in self.chunks.items():
    #         content = chunk["content"]
    #         print(f"Inserting chunk {chunk_id} into database...")
    #         await self.DB.insert_new_text_chunk(chunk_id, content)

    



