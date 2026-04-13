import aiosqlite
import sqlite_vec
import numpy as np
import os
import json
from SQL_UTILS import user_cache_helpers
from TokenizerWrapper.SmartTokenizer import Tokenizer
from OpenRouterWrapper.EmbeddingModel import OpenRouterEmbedding
import traceback
import asyncio

class AsyncVecSQLite:
    """
    AsyncVecSQLite is an asynchronous wrapper around a SQLite database connection
    that loads the sqlite_vec extension for vector operations.
    """
    def __init__(self, db_name, DIM=1024, tokenizer_model="meta-llama/llama-3.1-8b-instruct", embedding_model="baai/bge-large-en-v1.5", verbose=False):
        """
        Docstring for __init__ 
        :path is the path to the SQLite database file.
        :conn is the aiosqlite.Connection object.
        """
        self.path = f"Database/{db_name}.db"
        self.db_name = db_name
        self.conn: aiosqlite.Connection | None = None
        self.tokenizer = Tokenizer(model_name=tokenizer_model)
        self.EMBEDDING_DIM = DIM
        self.embedding_model = OpenRouterEmbedding(api_key=os.getenv("OPENROUTER_API_KEY"), model=embedding_model)
        self.verbose = verbose
        self.lock = asyncio.Lock()

    async def __aenter__(self):
        await self.open()
        await self.initialize_base_tables()
        return self

    async def open(self):
        """
        Docstring for open
        
        :Sets up the database connection and loads the sqlite_vec extension.
        """
        self.conn = await aiosqlite.connect(self.path)

        def setup():
            """
            Docstring for setup
            :Loads the sqlite_vec extension and sets PRAGMA settings for performance.
            :setup is synchronous because aiosqlite requires synchronous functions for execution.
            """
            raw = self.conn._conn  # sqlite3.Connection (DB thread safe here)

            raw.enable_load_extension(True)
            sqlite_vec.load(raw)
            raw.enable_load_extension(False)

            vec_version, = raw.execute(
                "SELECT vec_version()"
            ).fetchone()
            print(f"vec_version={vec_version}")

            raw.execute("PRAGMA journal_mode = WAL;")
            raw.execute("PRAGMA busy_timeout = 10000;")
            raw.execute("PRAGMA synchronous = OFF;")
            raw.execute("PRAGMA temp_store = MEMORY;")
            raw.execute("PRAGMA mmap_size = 30000000000;")
            raw.execute("PRAGMA cache_size = -200000;")
            raw.execute("PRAGMA foreign_keys = ON;")

        await self.conn._execute(setup)
        return self
    
    async def get_tables(self) -> list[str]:
        """
        This function retrieves the names of all user-defined tables in the SQLite database, excluding internal SQLite tables.
        """
        if not self.conn:
            raise RuntimeError("Database not opened")

        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
        """
        async with self.conn.execute(query) as cursor:
            rows = await cursor.fetchall()

        return [row[0] for row in rows]
    
    async def initialize_base_tables(self, schema="SQL_UTILS/SCHEMA.sql"):
        """
        Given a path to a SQL schema file, this function initializes the base tables required for the application, including tables for entities, text chunks, and their vector embeddings. It also creates virtual tables for efficient vector search using the sqlite_vec extension.
        """
        
        if schema is None:
            raise ValueError("Schema file path must be provided")
        
        if not self.conn:
            raise RuntimeError("Database not opened")
        
        with open(schema, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        if schema_sql.strip() == "":
            raise ValueError("Schema file is empty")

        ENTITY_VEC_TABLE_SCHEMA = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING vec0(
            id TEXT,
            embedding float[{self.EMBEDDING_DIM}]
        )
        """

        TEXT_CHUNK_VEC_TABLE_SCHEMA = f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS text_chunks_vec USING vec0(
            chunk_id TEXT,
            embedding float[{self.EMBEDDING_DIM}]
        )
        """
        await self.conn.executescript(schema_sql)
        await self.conn.execute(ENTITY_VEC_TABLE_SCHEMA)
        await self.conn.execute(TEXT_CHUNK_VEC_TABLE_SCHEMA)
        await self.conn.commit()

    async def insert_documents(self):
        pass

    async def does_entity_have_shadow(self, entity_id):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT 1 FROM shadow_entity_pair
        WHERE entity_id = ?
        LIMIT 1
        """

        async with self.conn.execute(SQL, (entity_id,)) as cursor:
            row = await cursor.fetchone()

        return row is not None

    async def insert_shadow_entity_pair(self, entity_id, shadow_id):
        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")

            SQL = """
            INSERT INTO shadow_entity_pair (shadow_id, entity_id)
            VALUES (?, ?)
            """

            await self.conn.execute(SQL, (shadow_id, entity_id))
            await self.conn.commit()

    async def does_shadow_id_exist(self, shadow_id):
            if self.conn is None:
                raise RuntimeError("Database not opened")

            SQL = """
            SELECT 1 FROM shadow_entity
            WHERE shadow_id = ?
            LIMIT 1
            """

            async with self.conn.execute(SQL, (shadow_id,)) as cursor:
                row = await cursor.fetchone()

            return row is not None

    async def insert_shadow_entity(self, shadow_id):
        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")

            SQL = """
            INSERT INTO shadow_entity (shadow_id)
            VALUES (?)
            """

            await self.conn.execute(SQL, (shadow_id,))
            await self.conn.commit()

    async def link_entity_to_shadow(self, entity_id, shadow_id):
        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")

            SQL = """
            INSERT INTO shadow_entity_to_entity (shadow_id, entity_id)
            VALUES(?,?)
            """

            await self.conn.execute(SQL, (shadow_id, entity_id))
            await self.conn.commit()



    async def insert_new_text_chunk(self, chunk_id, content):
        """
        Inserts a text chunk into the database along with its vector embedding. The chunk is stored in the 'text_chunks' table, while the embedding is stored in the 'text_chunks_vec' virtual table for efficient vector search.
        """

        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")
            
            SQL = """
            INSERT OR IGNORE INTO text_chunks (chunk_id, content)
            VALUES (?, ?)
            """

            cursor = await self.conn.execute(SQL, (chunk_id, content))

            # If the row count returns 0, it already exists, so we skip embedding and vector insertion to save resources
            if cursor.rowcount == 0:
                return  

            # However, if it is a new chunk, we proceed to generate its embedding and insert it into the vector table
            embedding = await self.embedding_model.embed(content)
            emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()

            VEC_SQL = """
            INSERT INTO text_chunks_vec (chunk_id, embedding)
            VALUES (?, ?)
            """

            await self.conn.execute(VEC_SQL, (chunk_id, emb_blob))
            await self.conn.commit()

    async def insert_new_marked_chunk(self, chunk_id, content):
        """
        Inserts a text chunk into the database along with its vector embedding. The chunk is stored in the 'text_chunks' table, while the embedding is stored in the 'text_chunks_vec' virtual table for efficient vector search.
        """

        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")
            
            SQL = """
            INSERT OR IGNORE INTO marked_chunks (chunk_id, content)
            VALUES (?, ?)
            """

            cursor = await self.conn.execute(SQL, (chunk_id, content))
            await self.conn.commit()

    async def insert_new_shadow_chunk(self, chunk_id, content):
        """
        Inserts a text chunk into the database along with its vector embedding. The chunk is stored in the 'text_chunks' table, while the embedding is stored in the 'text_chunks_vec' virtual table for efficient vector search.
        """

        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")
            
            SQL = """
            INSERT OR IGNORE INTO shadow_entity_chunks (chunk_id, content)
            VALUES (?, ?)
            """

            cursor = await self.conn.execute(SQL, (chunk_id, content))
            await self.conn.commit()

    async def insert_new_shadow_entity_chunk(self, chunk_id, content):
        """
        Inserts a text chunk into the database along with its vector embedding. The chunk is stored in the 'text_chunks' table, while the embedding is stored in the 'text_chunks_vec' virtual table for efficient vector search.
        """

        async with self.lock:
            if self.conn is None:
                raise RuntimeError("Database not opened")
            
            SQL = """
            INSERT OR IGNORE INTO shadow_entity_chunks (chunk_id, content)
            VALUES (?, ?)
            """

            cursor = await self.conn.execute(SQL, (chunk_id, content))
            await self.conn.commit()



    async def insert_entities(self, entity_info):
        import json
        import asyncio
        import numpy as np

        async with self.lock:

            if self.conn is None:
                raise RuntimeError("Database not opened")

            CHECK_PAIR_SQL = """
            SELECT 1 FROM entity_chunk_pairs
            WHERE entity_id = ? AND chunk_id = ?
            LIMIT 1
            """

            CHECK_ENTITY_SQL = """
            SELECT 1 FROM entities WHERE id = ?
            LIMIT 1
            """

            INSERT_ENTITY_SQL = """
            INSERT INTO entities (id, ename, etype, edesc)
            VALUES (?, ?, ?, ?)
            """

            UPDATE_ENTITY_SQL = """
            UPDATE entities
            SET edesc = edesc || '<SEP>' || ?
            WHERE id = ?
            """

            INSERT_PAIR_SQL = """
            INSERT INTO entity_chunk_pairs (entity_id, chunk_id)
            VALUES (?, ?)
            """

            VEC_INSERT_SQL = """
            INSERT INTO entities_vec (id, embedding)
            VALUES (?, ?)
            """

            to_embed = []

            for entity_name, data in entity_info.items():
                entity_id = data["id"]
                desc = data["desc"]
                etype = data["type"]
                chunk_id = data["chunk_id"]

                # 🔴 1. check pair FIRST
                async with self.conn.execute(CHECK_PAIR_SQL, (entity_id, chunk_id)) as cursor:
                    pair_exists = await cursor.fetchone()

                if pair_exists:
                    continue  # ✅ NOTHING happens

                # 🔴 2. only now check entity
                async with self.conn.execute(CHECK_ENTITY_SQL, (entity_id,)) as cursor:
                    entity_exists = await cursor.fetchone()

                if entity_exists:
                    # update ONLY because it's a new chunk
                    await self.conn.execute(UPDATE_ENTITY_SQL, (desc, entity_id))
                else:
                    # new entity
                    await self.conn.execute(
                        INSERT_ENTITY_SQL,
                        (entity_id, entity_name, etype, desc)
                    )

                # 🔴 3. insert pair (always for new pair)
                await self.conn.execute(INSERT_PAIR_SQL, (entity_id, chunk_id))

                # 🔴 4. embed (only for new work)
                to_embed.append((entity_name, data))

            async def embed_entity(entity_name, data):
                payload = {
                    "name": entity_name,
                    "desc": data["desc"],
                    "type": data["type"]
                }
                embedding = await self.embedding_model.embed(json.dumps(payload))
                return data["id"], np.asarray(embedding, dtype=np.float32).tobytes()

            results = await asyncio.gather(*[
                embed_entity(entity_name, data)
                for entity_name, data in to_embed
            ])

            for entity_id, emb_blob in results:
                await self.conn.execute(VEC_INSERT_SQL, (entity_id, emb_blob))

            await self.conn.commit()

    async def does_chunk_id_exist(self, chunk_id):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT 1 FROM text_chunks
        WHERE chunk_id = ?
        LIMIT 1
        """

        async with self.conn.execute(SQL, (chunk_id,)) as cursor:
            row = await cursor.fetchone()

        return row is not None


    # async def insert_entities(self, entity_info):
    #     """
    #     Batch inserts entities into the entities table.
    #     If an entity already exists (same id), append the new description
    #     using <SEP> instead of overwriting.

    #     Embeds each entity individually as a JSON object using async gather
    #     and inserts into entities_vec.
    #     """

    #     import json
    #     import asyncio
    #     import numpy as np

    #     if self.conn is None:
    #         raise RuntimeError("Database not opened")

    #     SELECT_SQL = """
    #     SELECT edesc FROM entities WHERE id = ?
    #     """

    #     INSERT_SQL = """
    #     INSERT INTO entities (id, ename, etype, edesc)
    #     VALUES (?, ?, ?, ?)
    #     """

    #     UPDATE_SQL = """
    #     UPDATE entities
    #     SET edesc = ?
    #     WHERE id = ?
    #     """

    #     VEC_INSERT_SQL = """
    #     INSERT INTO entities_vec (id, embedding)
    #     VALUES (?, ?)
    #     """

    #     # -------- TEXT TABLE (same logic as before) --------
    #     for entity_name, data in entity_info.items():
    #         entity_id = data["id"]
    #         desc = data["desc"]
    #         etype = data["type"]

    #         async with self.conn.execute(SELECT_SQL, (entity_id,)) as cursor:
    #             row = await cursor.fetchone()

    #         if row:
    #             existing_desc = row[0]
    #             if desc not in existing_desc:
    #                 new_desc = f"{existing_desc}<SEP>{desc}"
    #                 await self.conn.execute(UPDATE_SQL, (new_desc, entity_id))
    #         else:
    #             await self.conn.execute(
    #                 INSERT_SQL,
    #                 (entity_id, entity_name, etype, desc)
    #             )

    #     # -------- VECTOR TABLE (async gather, JSON input) --------
    #     async def embed_entity(entity_name, data):
    #         payload = {
    #             "name": entity_name,
    #             "desc": data["desc"],
    #             "type": data["type"]
    #         }
    #         embedding = await self.embedding_model.embed(json.dumps(payload))
    #         return data["id"], np.asarray(embedding, dtype=np.float32).tobytes()

    #     tasks = [
    #         embed_entity(entity_name, data)
    #         for entity_name, data in entity_info.items()
    #     ]

    #     results = await asyncio.gather(*tasks)

    #     for entity_id, emb_blob in results:
    #         await self.conn.execute(VEC_INSERT_SQL, (entity_id, emb_blob))

    #     await self.conn.commit()
    async def insert_shadow_chunk(self, chunk_id, content):
        """
        Inserts a shadow chunk into the database.
        """
        if self.conn is None:
            raise RuntimeError("Database not opened")
        
        SQL = """
        INSERT OR IGNORE INTO shadow_chunks (chunk_id, shadow_content)
        VALUES (?, ?)
        """
        await self.conn.execute(SQL, (chunk_id, content))
        await self.conn.commit()


    async def get_all_shadow_entity_chunks(self):
        """
        Retrieves a shadow chunk from the database by its chunk_id.
        """
        if self.conn is None:
            raise RuntimeError("Database not opened")
        
        SQL = """
        SELECT chunk_id, content FROM shadow_entity_chunks
        """

        chunk_id_to_text = {}

        async with self.conn.execute(SQL) as cursor:
            rows = await cursor.fetchall()

       
        for chunk_id, text in rows:
            chunk_id_to_text[chunk_id] = text

        return chunk_id_to_text
    
    async def get_shadow_chunk(self, chunk_id):
        """
        Retrieves a shadow chunk from the database by its chunk_id.
        """
        if self.conn is None:
            raise RuntimeError("Database not opened")
        
        SQL = """
        SELECT content FROM shadow_chunks
        WHERE chunk_id = ?;
        """
        async with self.conn.execute(SQL, (chunk_id,)) as cursor:
            row = await cursor.fetchone()

        return row[0] if row else None


   
    async def gather_entities_from_given_chunk(self, chunk_id, allowed_types=None):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        async with self.lock:

            entity_to_chunk_table = "entity_chunk_pairs"

            if allowed_types:
                placeholders = ",".join("?" for _ in allowed_types)

                sql = f"""
                SELECT e.ename
                FROM {entity_to_chunk_table} ec
                JOIN entities e ON e.id = ec.entity_id
                WHERE e.etype IN ({placeholders})
                AND ec.chunk_id = ?
                """

                params = (*allowed_types, chunk_id)

            else:
                sql = f"""
                SELECT e.ename
                FROM {entity_to_chunk_table} ec
                JOIN entities e ON e.id = ec.entity_id
                WHERE ec.chunk_id = ?
                """

                params = (chunk_id,)

            async with self.conn.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        return [row[0] for row in rows]

    async def semantic_search_over_text_chunks(self, query, top_k=40):
        if self.conn is None:
            raise RuntimeError("Database not opened")
        
        emb_blob = await user_cache_helpers.prepare_embedding_for_query(
            self.conn,
            self.embedding_model.embed,
            query
        )

        chunk_table = "text_chunks"
        vec_table = "text_chunks_vec"

        SQL = f"""
        SELECT
            c.chunk_id,
            c.content,
            vec_distance_L2(v.embedding, ?) AS distance
        FROM {chunk_table} c
        JOIN {vec_table} v ON v.chunk_id = c.chunk_id
        ORDER BY distance ASC
        LIMIT ?
        """

        async with self.conn.execute(SQL, (emb_blob, top_k)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "content": row[1],
                "distance": row[2],
            }
            for row in rows
        ]
    
    async def map_chunks_to_shadow_chunks(self, chunk_dict):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        # extract chunk_ids
        chunk_ids = [c["chunk_id"] for c in chunk_dict]

        if not chunk_ids:
            return []

        placeholders = ",".join("?" for _ in chunk_ids)

        SQL = f"""
        SELECT chunk_id, shadow_content
        FROM shadow_chunks
        WHERE chunk_id IN ({placeholders})
        """

        async with self.conn.execute(SQL, chunk_ids) as cursor:
            rows = await cursor.fetchall()

        # map chunk_id → shadow_content
        shadow_map = {row[0]: row[1] for row in rows}

        # merge back into original structure
        result = []
        for c in chunk_dict:
            result.append({
                **c,
                "shadow_chunk": shadow_map.get(c["chunk_id"])  # None if missing
            })

        return result
        
    
    
    # async def semantic_search_over_text_chunks(self, query, eid, top_k=10):
    #         if self.conn is None:
    #             raise RuntimeError("Database not opened")
            
    #         emb_blob = await user_cache_helpers.prepare_embedding_for_query(
    #             self.conn,
    #             self.embedding_model.embed,
    #             query
    #         )

    #         chunk_table = "text_chunks"
    #         entity_to_chunk_table = f"entity_chunk_pairs"

    #         SQL = f"""
    #         SELECT
    #             c.chunk_id,
    #             c.content,
    #             vec_distance_L2(c.embedding, ?) AS distance
    #         FROM {chunk_table} c
    #         JOIN {entity_to_chunk_table} ec
    #             ON ec.chunk_id = c.chunk_id
    #         WHERE ec.entity_id = ?
    #         ORDER BY distance ASC
    #         LIMIT ?
    #         """

    #         async with self.conn.execute(SQL, (emb_blob, eid, top_k)) as cursor:
    #             rows = await cursor.fetchall()

    #         return [
    #             {
    #                 "chunk_id": row[0],
    #                 "content": row[1],
    #                 "distance": row[2],
    #             }
    #             for row in rows
    #         ]
    
    async def entity_type_semantic_search(self, query, top_k=10):
        if self.conn is None:
                raise RuntimeError("Database not opened")
        
        table = "type_cache"
        emb_blob = await user_cache_helpers.prepare_embedding_for_query(
                self.conn,
                self.embedding_model.embed,
                query,
                table=table
            )
        
        SQL = f"""
            SELECT
                t.query_hash,
                t.type_desc,
                vec_distance_L2(t.embedding, ?) AS distance
            FROM {table} t
            ORDER BY distance ASC
            LIMIT ?
            """
        
        async with self.conn.execute(SQL, (emb_blob, top_k)) as cursor:
                rows = await cursor.fetchall()

                return [
                {
                    "type_id": row[0],
                    "content": row[1],
                    "distance": row[2],
                }
                for row in rows
            ]
       

    async def naive_semantic_search(self, query, top_k=10):
        if self.conn is None:
            raise RuntimeError("Database not opened")
        
        emb_blob = await user_cache_helpers.prepare_embedding_for_query(
            self.conn,
            self.embedding_model.embed,
            query
        )

        chunk_table = "text_chunks"

        SQL = f"""
        SELECT
            c.chunk_id,
            c.content,
            vec_distance_L2(c.embedding, ?) AS distance
        FROM {chunk_table} c
        ORDER BY distance ASC
        LIMIT ?
        """

        async with self.conn.execute(SQL, (emb_blob, top_k)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "content": row[1],
                "distance": row[2],
            }
            for row in rows
        ]


    async def compare_two_entity_embeddings(self, src_id, dst_id):
        if not self.conn:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT vec_distance_L2(a.embedding, b.embedding) AS distance
        FROM entities_vec a
        JOIN entities_vec b
            ON a.id = ?
        AND b.id = ?
        """

        async with self.conn.execute(SQL, (src_id, dst_id)) as cursor:
            row = await cursor.fetchone()

        return row[0] if row else None


    async def global_semantic_search_over_all_entities(self, query, top_k=1):

        if not self.conn:
            raise RuntimeError("Database not opened")

        emb_blob = await user_cache_helpers.prepare_embedding_for_query(
            self.conn,
            self.embedding_model.embed,
            query
        )

        SQL = """
        SELECT
            e.id,
            e.ename,
            e.etype,
            e.edesc,
            v.distance
        FROM entities_vec v
        JOIN entities e ON e.id = v.id
        WHERE v.embedding MATCH ?
        AND k = ?
        ORDER BY v.distance ASC
        """

        async with self.conn.execute(SQL, (emb_blob, top_k)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "description": row[3],
                "distance": row[4],
            }
            for row in rows
        ]
    

    async def compare_entity_embedding_with_query(self, entity_name, query):
        if not self.conn:
            raise RuntimeError("Database not opened")

        emb_blob = await user_cache_helpers.prepare_embedding_for_query(
            self.conn,
            self.embedding_model.embed,
            query
        )

        SQL = """
        SELECT
            vec_distance_L2(v.embedding, ?) AS distance
        FROM entities_vec v
        JOIN entities e ON e.id = v.id
        WHERE e.ename = ?
        """

        async with self.conn.execute(
            SQL,
            (emb_blob, entity_name)
        ) as cursor:
            row = await cursor.fetchone()

        return row[0] if row else None


    async def global_semantic_search_over_all_entities(self, query, top_k=1):

        if not self.conn:
            raise RuntimeError("Database not opened")

        emb_blob = await user_cache_helpers.prepare_embedding_for_query(
            self.conn,
            self.embedding_model.embed,
            query
        )

        SQL = """
        SELECT
            e.id,
            e.ename,
            e.etype,
            e.edesc,
            v.distance
        FROM entities_vec v
        JOIN entities e ON e.id = v.id
        WHERE v.embedding MATCH ?
        AND k = ?
        ORDER BY v.distance ASC
        """

        async with self.conn.execute(SQL, (emb_blob, top_k)) as cursor:
            rows = await cursor.fetchall()

        return [
            {
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "description": row[3],
                "distance": row[4],
            }
            for row in rows
        ]
    
    async def get_entity_info(self,entity_id):
        SQL = """
        SELECT ename, etype, edesc
        FROM entities
        WHERE id = ?
        LIMIT 1
        """

        async with self.conn.execute(SQL, (entity_id,)) as cursor:
            row = await cursor.fetchone()

        ename, etype, edesc = row

        return {
            "id": entity_id,
            "name": ename,
            "type": etype,
            "desc": edesc
        }

    async def separate_cluster_by_type(self, cluster_data):
        type_groups = {}

        for entity in cluster_data:
            etype = entity["type"]

            if etype not in type_groups:
                type_groups[etype] = []

            type_groups[etype].append(entity)

        # return list of clusters (grouped by type)
        return list(type_groups.values())
    

    async def clusters_to_json_strings(self, clusters):
        """
        Takes a list of clusters (list of lists).
        For each cluster:
            - fetch descriptions concurrently using get_entity_desc
            - build JSON string with fields:
                id, entity_name, desc
        Returns:
            List[str] (one JSON string per cluster)
        """

        cluster_json_strings = []

        for cluster in clusters:
            # Fetch descriptions concurrently for this cluster
            cluster_data = await asyncio.gather(
                *[self.get_entity_info(entity_id) for entity_id in cluster]
            )

            # Slice by types
            separated_clusters = await self.separate_cluster_by_type(cluster_data)

            for subcluster in separated_clusters:
                if len(subcluster) <= 1:
                    continue

            cluster_json_strings.append(json.dumps(subcluster))

        return cluster_json_strings

    async def insert_action_chunk_pair(self, chunk_id, action_span, shadow_action):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        INSERT OR IGNORE INTO action_chunk_pairs (action_span, chunk_id, shadow_action)
        VALUES (?, ?, ?)
        """

        await self.conn.execute(SQL, (action_span, chunk_id, shadow_action))
        await self.conn.commit()

    async def get_action_chunk_pair(self, chunk_id, shadow_action):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT action_span, chunk_id, shadow_action
        FROM action_chunk_pairs
        WHERE chunk_id = ? AND shadow_action = ?
        LIMIT 1
        """

        async with self.conn.execute(SQL, (chunk_id, shadow_action)) as cursor:
            row = await cursor.fetchone()

        action_span, chunk_id, shadow_action = row

        return json.dumps({
            "chunk_id": chunk_id,
            "action_span": action_span,
            "shadow_action": shadow_action
        })

    async def get_chunk_real_and_shadow_entities(self, chunk_id):

        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT
            ecp.chunk_id,
            e.id AS entity_id,
            e.ename AS entity_name,
            sete.shadow_id AS shadow_id,
            sete.shadow_id AS shadow_name
        FROM entity_chunk_pairs ecp
        JOIN entities e ON e.id = ecp.entity_id
        LEFT JOIN shadow_entity_to_entity sete ON sete.entity_id = e.id
        WHERE ecp.chunk_id = ?
        """

        async with self.conn.execute(SQL, (chunk_id,)) as cursor:
            rows = await cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "entity_id": row[1],
                "entity_name": row[2],
                "shadow_id": row[3],
                "shadow_name": row[4],
            })

        return results
        
    async def get_all_marked_chunks(self):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT chunk_id, content FROM marked_chunks
        """

        chunk_id_to_text = {}

        async with self.conn.execute(SQL) as cursor:
            rows = await cursor.fetchall()

        # build matrix
        for chunk_id, text in rows:
            chunk_id_to_text[chunk_id] = text

        return chunk_id_to_text


            





    async def cluster_similar_entities(self, threshold=0.65):
        if self.conn is None:
            raise RuntimeError("Database not opened")

        SQL = """
        SELECT id, embedding FROM entities_vec
        """

        entity_ids = []
        embeddings = []

        async with self.conn.execute(SQL) as cursor:
            rows = await cursor.fetchall()

        # build matrix
        for entity_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32)
            entity_ids.append(entity_id)
            embeddings.append(vec)

        if not embeddings:
            return []

        embedding_matrix = np.vstack(embeddings)

        # normalize (cosine trick)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        embedding_matrix = embedding_matrix / norms

        # cosine similarity via dot product
        similarity_matrix = embedding_matrix @ embedding_matrix.T

        visited = set()
        duplicate_clusters = []
        N = len(entity_ids)

        for i in range(N):
            if i in visited:
                continue

            similar_indices = np.where(similarity_matrix[i] >= threshold)[0]
            similar_indices = [idx for idx in similar_indices if idx != i]

            if similar_indices:
                cluster = [entity_ids[i]] + [entity_ids[idx] for idx in similar_indices]
                duplicate_clusters.append(cluster)

                visited.update(similar_indices)
                visited.add(i)

        return await self.clusters_to_json_strings(duplicate_clusters), entity_ids   
    

    async def create_temp_vector_table(self):
        pass

    async def semantic_search(self):
        pass

    async def close(self):
        if self.conn:
            await self.conn.close()
            self.conn = None

        if self.embedding_model:
         await self.embedding_model.close()
            
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()