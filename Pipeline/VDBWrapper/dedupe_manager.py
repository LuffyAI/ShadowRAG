import json
import hashlib
import asyncio
import random
from itertools import combinations
from collections import defaultdict
import re


class DedupeManager:
    def __init__(self, clusters, entity_ids, semaphore, max_tries, DB, llm_func):
        self.clusters = clusters
        self.entity_ids = set(entity_ids)
        self.merge_choices = None
        self.single_choices = None
        self.MAX_RETRIES = max_tries
        self.DB = DB
        self.semaphore = semaphore  
        self.call_llm_func = llm_func

    async def get_marked_chunks(self):
        return await self.DB.get_all_marked_chunks()

    async def swap_entities_with_shadow(self, chunk_id, chunk_text):

        mod_json = await self.DB.get_chunk_real_and_shadow_entities(chunk_id)

        text = chunk_text.lower()

        for row in mod_json:
            entity_name = row["entity_name"]
            shadow_name = row["shadow_name"]

            if not shadow_name:
                continue  # skip if no shadow

            entity_lower = entity_name.lower()

            # match **[entity]**
            pattern = re.compile(rf"\*\*\[{re.escape(entity_lower)}\]\*\*")

            replacement = f"**[{shadow_name}]**"

            text = pattern.sub(replacement, text)
        await self.DB.insert_new_shadow_entity_chunk(chunk_id, text)

        return text

    async def batch_entity_swapping(self):

            chunk_dict = await self.get_marked_chunks()  # {chunk_id: text}
            results = {}

            async def process_single(chunk_id, chunk_text):
                async with self.semaphore:
                    new_text = await self.swap_entities_with_shadow(chunk_id, chunk_text)
                    return chunk_id, new_text

            tasks = [
                process_single(chunk_id, chunk_text)
                for chunk_id, chunk_text in chunk_dict.items()
            ]

            outputs = await asyncio.gather(*tasks)

            for chunk_id, new_text in outputs:
                results[chunk_id] = new_text



    async def batch_singleton_shadows(self):
        # create tasks for all remaining entity_ids
        tasks = [
            self.get_singleton_shadow_id(entity_id)
            for entity_id in self.entity_ids
        ]

        # run concurrently
        shadow_ids = await asyncio.gather(*tasks)

        # build mapping: shadow_id -> original entity_id
        singleton_shadow_dict = {}

        for entity_id, shadow_id in zip(self.entity_ids, shadow_ids):
            singleton_shadow_dict[shadow_id] = entity_id

        return singleton_shadow_dict

    async def get_singleton_shadow_id(self, entity_id):
        # call get entity info from database

        entity_json = await self.DB.get_entity_info(entity_id)
        while True:
            shadow_id = f"{entity_json['type']}-{random.randint(0, 99999)}"
            exists = await self.DB.does_shadow_id_exist(shadow_id)

            if not exists:
                await self.DB.insert_shadow_entity(shadow_id)
                await self.DB.link_entity_to_shadow(entity_id=entity_id, shadow_id=shadow_id)
                break

        return shadow_id

    async def batch_merge_choices(self):
        async def process_single(cluster_json, idx):
            tries = 0

            while tries < self.MAX_RETRIES:
                try:
                    async with self.semaphore:
                        result = await self.make_dedupe_choice(cluster_json)
                    return result

                except Exception as e:
                    tries += 1
                    print(f"[Cluster {idx}] Retry {tries}: {e}")

                    if tries < self.MAX_RETRIES:
                        await asyncio.sleep(20)
                    else:
                        print(f"[Cluster {idx}] FAILED after {self.MAX_RETRIES} tries")
                        return None

        tasks = [
            process_single(cluster_json, idx)
            for idx, cluster_json in enumerate(self.clusters)
        ]

        batch_results = await asyncio.gather(*tasks)

        for res in batch_results:
            if res:
                self.merge_choices.append(res)



    async def prepare_cluster_json_for_llm(self, cluster_json):
        # parse input JSON string → list[dict]
        cluster = json.loads(cluster_json)
        etype = None

        id_map = {}
        new_cluster = []

        # This sets an id to each entity in a cluster, mapped alongside its entity name and description.
        for idx, entity in enumerate(cluster, start=1):
            original_id = entity["id"]
            etype = entity["type"]
            new_id = idx

            # build mapping (new → original)
            id_map[new_id] = original_id

            # copy entity and replace id
            new_entity = entity.copy()
            new_entity["id"] = new_id

            # remove type key from
            del new_entity["type"]

            # rename name to entity name
            new_entity["entity_name"] = new_entity.pop("name")

            # rename desc to entity_description
            new_entity["entity_description"] = new_entity.pop("desc")
            new_cluster.append(new_entity)

    
        #Prepare questions (First, we need to figure out how many unique combinations there exist)
        # (1,2,3) then we have (1,2), (1,3) and (2,3)
        pair_questions = {}
        pair_map = {}        # number -> (id_a, id_b)
        reverse_pair_map = {}  # (id_a, id_b) -> number
        sample_answer_dict = {}


        for idx, (a, b) in enumerate(combinations(new_cluster, 2), start=1):
            id_a = a["id"]
            id_b = b["id"]

            question = (
                f"Is '{a['entity_name']}' (description: {a['entity_description']}) "
                f"an alias of '{b['entity_name']}' "
                f"(description: {b['entity_description']})? "
            )

            pair_questions[idx] = question
            pair_map[idx] = (id_a, id_b)
            reverse_pair_map[(id_a, id_b)] = idx
            sample_answer_dict[idx] = "<Write your answer here>"

        print(pair_questions)

        return pair_questions, pair_map, sample_answer_dict, id_map, etype

    async def make_dedupe_choice(self, cluster_json):

        print("Entered")

        new_cluster_json, pair_map, sample_answer_dict, id_map, etype = await self.prepare_cluster_json_for_llm(cluster_json)

        PROMPT = f"""
        <Role>
        You will answer the following {len(new_cluster_json.items())} yes or no questions. Read them carefully and decide if it is a YES(1) or a NO (0).

        <QUESTIONS YOU MUST ANSWER>
        {new_cluster_json}
        </QUESTIONS YOU MUST ANSWER>


         <Output Format>
        - Return ONLY valid JSON of the following format.
        - The key is the question id and the value is your choice 1 or 0.
        - DO NOT return anything outside JSON.
        {sample_answer_dict}
        </Output Format>
        """
        

        print("PROMPT", PROMPT)

        entity_dict = json.loads(await self.call_llm_func(
        messages=[{"role": "user", "content": PROMPT}]
        ))

        print("merge choice", entity_dict)

        # For each number in the value, we convert it back to their real entity_ids with id_maps
        remapped_entity_dict = {}
        print("pair map", pair_map)
        print("id_map", id_map)

        # collect edges first
        edges = []
        nodes = set()

        for q_id_str, answer in entity_dict.items():
            if answer != 1:
                 continue

            q_id = int(q_id_str)

            # step 1: get local ids (e.g., 1,2)
            local_a, local_b = pair_map[q_id]

            # step 2: map to real entity ids
            real_a = id_map[local_a]
            real_b = id_map[local_b]

            edges.append((real_a, real_b))
            nodes.add(real_a)
            nodes.add(real_b)

        # union-find setup
        parent = {}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        # initialize parents
        for node in nodes:
            parent[node] = node

        # union all edges
        for a, b in edges:
            union(a, b)

        # group into components
        groups = defaultdict(list)
        for node in nodes:
            root = find(node)
            groups[root].append(node)

        # build final shadow entities
        remapped_entity_dict = {}

        for group in groups.values():
            while True:
                shadow_id = f"{etype}-{random.randint(0, 99999)}"
                exists = await self.DB.does_shadow_id_exist(shadow_id)

                if not exists:
                    await self.DB.insert_shadow_entity(shadow_id)
                    break

            remapped_entity_dict[shadow_id] = group

    
        # remove grouped entity_ids from the master set
        for shadow_id, group in remapped_entity_dict.items():
            for entity_id in group:
                await self.DB.link_entity_to_shadow(entity_id=entity_id, shadow_id=shadow_id)
                self.entity_ids.discard(entity_id)

        print("Remapped", remapped_entity_dict)
        print("entity set", self.entity_ids)
        return remapped_entity_dict






   