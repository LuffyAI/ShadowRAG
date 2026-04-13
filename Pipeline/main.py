import argparse
import json
import asyncio
from VDBWrapper.chunk_manager import ChunkManager
from VDBWrapper.construction_ws import AsyncShadowRAGConstructionWorkspace as ConstructionWS
from VDBWrapper.dedupe_ws import AsyncNormalizeEntitiesWorkspace as NE
from VDBWrapper.action_ws import AsyncActionWorkspace as AW


from OpenRouterWrapper.ChatBot import OpenRouterChat
from TokenizerWrapper.SmartTokenizer import Tokenizer
from SQL_UTILS.SQLITE_WRAPPER import AsyncVecSQLite as SQLiteWrapper
from dotenv import load_dotenv
import os
load_dotenv()

async def main(args):
    if args.chunk_documents:
        TokenizerWrapper = Tokenizer(model_name="meta-llama/llama-3.2-3b-instruct")
        embedding_model = OpenRouterEmbedding(api_key=os.getenv("OPENROUTER_API_KEY"), model="baai/bge-large-en-v1.5")
        chunk_manager = ChunkManager(document="Your document content here", tokenizer=TokenizerWrapper, EmbeddingModel=embedding_model)
        chunk_manager.chunk_text_with_ids()
        print(chunk_manager.chunks)
    elif args.embed_test:
        from OpenRouterWrapper.EmbeddingModel import OpenRouterEmbedding
        embedding = await embedding_model.embed("Your text to embed here")
        print(embedding)
        await embedding_model.close()
    elif args.dedupe:
        async with SQLiteWrapper(db_name="test") as DB:
            ans,X = await DB.cluster_similar_entities()
            print(ans)
            print(X)

    elif args.retrieval:
        pass

    elif args.test:
        from OpenRouterWrapper.EmbeddingModel import OpenRouterEmbedding

        data = {
         "PERSON": "A person specifically named in the text chunk, including fictional characters",
         "ORG": "An organization/group/company specifically named in the text chunk."
         }

        # async with ConstructionWS(
        #      db_name="test60",
        #     documents = "Larnell Moore purchased 400 billion dollars of pizza rolls.",
        #      sensitive_entity_dict=data,
        #  ) as workspace:
        #      pass

        # async with NE(
        #     db_name="test60",
        #     DIM=1024,
        #     model="meta-llama/llama-3.1-8b-instruct",
        #     embedding_model="baai/bge-large-en-v1.5",
        #     verbose=True
        # ) as workspace:
        #     pass

        sensitive_actions = {
        "companies": "Switch all companies to video game-related companies",
        "transactions": "Switch all memtions of money to gamestop award points"
        }

        replacement_actions = {
        "companies": "Obfuscate any token that reference a company, type of company, or industry by changing such tokens to a refer to an academic club or type of club.",
        "Transactions": "Obfuscate all transaction and money-related tokens. Keep the number and exact amount, but don't reveal that it is money.."
        }

        async with AW(
            db_name="test60",
            sensitive_action_dict=sensitive_actions,
            replacement_dict=replacement_actions,
            DIM=1024,
            model="meta-llama/llama-3.1-8b-instruct",
            embedding_model="baai/bge-large-en-v1.5",
            verbose=True
        ) as workspace:
            pass
        

    #    Chat = OpenRouterChat(api_key=os.getenv("OPENROUTER_API_KEY"), model="meta-llama/llama-3.1-8b-instruct")
    #    data = {
    #     "PERSON": "A person specifically named in the text chunk, including fictional characters",
    #     "ORG": "An organization/group/company specifically named in the text chunk."
    #     }

    #    entity_dict, types  = await extract_sensitive_entities_from_chunk(
    #         chunk="Elon Musk cofounded seven companies, including electric car maker Tesla, rocket producer SpaceX and artificial intelligence startup xAI.",
    #         sensitive_entity_classes=data,
    #         call_llm_func=Chat.generate
    #     )
       
    #    out2 = await create_entity_descriptions(
    #     chunk="Elon Musk cofounded seven companies, including electric car maker Tesla, rocket producer SpaceX and artificial intelligence startup xAI.",
    #     entity_dict=json.loads(entity_dict),
    #     type_id_to_name=types,
    #     call_llm_func=Chat.generate
    #      )

    #    print(out2)


    

    #    async with SQLiteWrapper(db_name="test") as DB:
    #        await DB.insert_entities(out2)
           
    #         # chunk_manager = ChunkManager(document="Your document content here", DB=DB)
    #         # print(await DB.get_tables())
    #         # await chunk_manager.insert_chunks_into_db()
    #         # await DB.insert_shadow_chunk(chunk_id="chunk-50395acf1dfcf8ed4c703923ecf188ab", content="This is a shadow chunk content")
    #         test = await DB.semantic_search_over_text_chunks(query="Document", top_k=5)
    #         print(test)
    #         cool = await DB.map_chunks_to_shadow_chunks(test)
    #         print(cool)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chunk_documents", action='store_true', help='Runs the DotRAG pipeline on the MetaQA dataset')
    group.add_argument("--embed_test", action='store_true', help='Tests the embedding functionality of the OpenRouterEmbedding class')
    group.add_argument("--test", action='store_true', help='Runs the test functionality')
    group.add_argument("--dedupe", action='store_true', help='Runs the deduplication workspace')
    group.add_argument("--retrieval", action='store_true', help='Tests the retrieval functionality of the database')
    args = parser.parse_args()
    asyncio.run(main(args))