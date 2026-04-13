import uuid
"""
we need a workspace to grab entities off each chunk
and managing construction

"""
class AsyncNeighborhoodWorkspace:
    def __init__(self, conn, eids, lock, verbose=False):
        self.conn = conn
        self.wid = uuid.uuid4().hex[:8]                 
        self.lock = lock
        self.verbose = verbose

    async def get_tables(self) -> list[str]:
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
    

    async def __aenter__(self):
        async with self.lock:
            for eid in self.eids:
                await self.conn.execute(f"""
                CREATE TEMP TABLE neighborhood_entities_{eid}_{self.wid} (
                    id INTEGER NOT NULL,
                    etype TEXT NOT NULL,
                    ename TEXT NOT NULL
                );
                """)

                await self.conn.execute(f"""
                CREATE TEMP TABLE neighborhood_relationships_{eid}_{self.wid} (
                    id_src INTEGER NOT NULL,
                    id_dst INTEGER NOT NULL,
                    src_name TEXT NOT NULL,
                    src_type TEXT NOT NULL,
                    dst_type TEXT NOT NULL,
                    dst_name TEXT NOT NULL,
                    rdesc TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    hop INTEGER NOT NULL
                );
                """)

                if self.verbose:
                 print(f"Created temporary neighborhood tables for eid {eid} with workspace id {self.wid}")
            return self

    async def __aexit__(self, exc_type, exc, tb):
        pass
        # async with self.lock:
        #     if self.verbose:
        #      print(f"Cleaning up temporary neighborhood tables for workspace id {self.wid}...")

        #     tables = await self.get_tables()

        #     for eid in self.eids:
        #         await self.conn.execute(
        #             f"DROP TABLE IF EXISTS neighborhood_entities_{eid}_{self.wid}"
        #         )
        #         await self.conn.execute(
        #             f"DROP TABLE IF EXISTS neighborhood_relationships_{eid}_{self.wid}"
        #         )

        #         prefix = f"neighborhood_vector_search_{eid}_{self.wid}"
        #         for table in tables:
        #             if table.startswith(prefix):
        #                 await self.conn.execute(f"DROP TABLE IF EXISTS {table}")

        #     await self.conn.commit()