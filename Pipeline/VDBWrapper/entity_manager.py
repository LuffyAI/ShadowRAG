import json
import hashlib
import asyncio
import re

class EntityManager:
    def __init__(self, chunks, semaphore, max_tries, sensitive_class_dict, DB, llm_func):
        self.chunks = chunks
        self.entity_batch = []
        self.entity_list = None
        self.MAX_RETRIES = max_tries
        self.sensitive_classes = sensitive_class_dict
        self.DB = DB
        self.semaphore = semaphore  # Limit concurrent LLM calls
        self.call_llm_func = llm_func

    async def process_entities(self):
        results = []

        async def process_single(chunk_str, chunk_id):
            # -------- STAGE 1: EXTRACT --------
            extract_retries = 0
            entity_dict = None
            number_to_type = None

            while extract_retries < self.MAX_RETRIES:
                try:
                    async with self.semaphore:
                        entity_dict, number_to_type, new_chunk_str = await self.extract_sensitive_entities_from_chunk(chunk_id, chunk_str)
                    break
                except Exception as e:
                    extract_retries += 1
                    print(f"[Extract Retry {extract_retries}] chunk {chunk_id}: {e}")

                    if extract_retries < self.MAX_RETRIES:
                        await asyncio.sleep(20)
                    else:
                        print(f"[FAILED EXTRACT] chunk {chunk_id}")
                        return None  # stop completely for this chunk

            # -------- STAGE 2: DESCRIPTIONS --------
            desc_retries = 0

            while desc_retries < self.MAX_RETRIES:
                try:
                    async with self.semaphore:
                        enriched = await self.create_entity_descriptions(
                            new_chunk_str,
                            chunk_id,
                            entity_dict,
                            number_to_type
                        )
                    return enriched
                except Exception as e:
                    desc_retries += 1
                    print(f"[Desc Retry {desc_retries}] chunk {chunk_id}: {e}")

                    if desc_retries < self.MAX_RETRIES:
                        await asyncio.sleep(20)
                    else:
                        print(f"[FAILED DESC] chunk {chunk_id}")
                        return None

        tasks = [
            process_single(chunk_str, chunk_id)
            for chunk_id, chunk_str in self.chunks.items()
        ]

        batch_results = await asyncio.gather(*tasks)

        for res in batch_results:
            if res:
                results.append(res)

        self.entity_list = results

    async def extract_from_one_chunk(self, chunk_str, chunk_id):
        entity_dict, number_to_type = await self.extract_sensitive_entities_from_chunk(chunk_id, chunk_str)
        return self.create_entity_descriptions(chunk_str, chunk_id, entity_dict, number_to_type)

    async def render_sensitive_classes(self, data):
    
        numbered_map = {
            i: {"label": key, "definition": value}
            for i, (key, value) in enumerate(data.items(), start=1)
        }

        formatted_string = "\n".join(
            f"{i}: {entry['label']} - {entry['definition']}"
            for i, entry in numbered_map.items()
        )

        return numbered_map, formatted_string
    

    async def mark_sensitive_entity_text_spans(self, chunk_id, chunk, entity_dict):
        print(chunk, "chunk")
        marked_chunk = chunk['content'].lower()
        filtered_entity_dict = {}

        for entity in entity_dict:
            print(entity_dict)
            lowercased_entity = entity.lower()
            pattern = re.compile(rf"\b{re.escape(lowercased_entity)}\b")

            if not pattern.search(marked_chunk):
                continue

            filtered_entity_dict[lowercased_entity] = entity_dict[entity]
            marked_chunk = pattern.sub(f"**[{lowercased_entity}]**", marked_chunk)

        chunk['content'] = marked_chunk

        await self.DB.insert_new_marked_chunk(chunk_id=chunk_id, content=marked_chunk)

        return filtered_entity_dict, chunk


    async def extract_sensitive_entities_from_chunk(self, chunk_id, chunk):

        number_to_type, _ = await self.render_sensitive_classes(self.sensitive_classes)

        PROMPT = f"""
        <Role>
        You are an assistant tasked with extracting all unique sensitive entities from the provided text chunk that meet the provided classification types. You will extract exact text spans. You are not inferring or predicting entities not stated directly in the passage. You are not creating new sensitive entity classes.
        </Role>

       <Instructions>
        1. Read the sensitive entity classes.
        2. Identify all named entities in the text chunk that meet the definition of one of the provided sensitive entity classifications. Extract as many as possible.
        3. Your answer should be a JSON where the key is the extracted sensitive named entity and its value is the number of its associated classification.
        </Instructions>

        <Allowed Sensitive Entity Classes>
        There exist {len(self.sensitive_classes.keys())} possible types an entity can take on. An entity can only be of one class. Keep this in mind.
        {number_to_type}
        </Allowed Sensitive Entity Classes>

        <Text Chunk>
        {chunk['content']}
        </Text Chunk>

        <Example Output Format>
        {{"Elon Musk": "1", "": "Telsa": "2", ...}}
        </Example Output Format>
        """

        entity_dict = json.loads(await self.call_llm_func(
        messages=[{"role": "user", "content": PROMPT}]
        ))

        print("PROMPT SENT", PROMPT)

        filtered_entity_dict, new_text_chunk = await self.mark_sensitive_entity_text_spans(chunk_id, chunk, entity_dict)
        print("New chunk", chunk)


        return filtered_entity_dict, number_to_type, chunk

    async def create_entity_descriptions(self, chunk, chunk_id, entity_dict, number_to_type):
   
        entity_with_labels = {
            entity: number_to_type[int(type_id)]["label"]
            for entity, type_id in entity_dict.items()
        }

        # helper for A / An
        def article(word):
            return "An" if word[0].lower() in "aeiou" else "A"

        # dynamically build example JSON
        example_json = "{\n" + ",\n".join(
            f'  "{entity}": "{article(label)} {label} that...[YOU MUST FILL IN THE REST]"'
            for entity, label in entity_with_labels.items()
        ) + "\n}"

        PROMPT = f"""
        <Role>
        You are an assistant tasked with writing a two-to-four sentence description of the given entities based on the text chunk.
        </Role>

        <Instructions>
        1. Observe the provided entity list and read the text chunk.
        2. Write a two-to-four sentence description of the provided entities based on your pre-trained knowledge and information in the text chunk.
        3. Your answer should be a JSON where the key is a provided entity and its value is its definition.
        </Instructions>

        <Provided Entities>
        You must write one definition each for the {len(entity_dict)} provided entities:
        {entity_with_labels}
        </Provided Entities>

        <Text Chunk>
        {chunk}
        </Text Chunk>

        <Example Output Format>
        {example_json}
        </Example Output Format>
        """

        print(PROMPT)

        response = json.loads(await self.call_llm_func(
            messages=[{"role": "user", "content": PROMPT}]
        ))

        print(response)

        enriched_response = {}

        for entity, desc in response.items():
            label = entity_with_labels.get(entity)

            # md5 hash
            entity_hash = hashlib.md5(entity.encode()).hexdigest()
            entity_id = f"entity-{entity_hash}"

            enriched_response[entity] = {
                "chunk_id": chunk_id,
                "id": entity_id,
                "desc": desc,
                "type": label
            }

        return enriched_response


