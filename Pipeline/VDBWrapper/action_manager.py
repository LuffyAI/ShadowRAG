import json
import hashlib
import asyncio
import re

class ActionManager:
    def __init__(self, semaphore, max_tries, sensitive_actions, replacements, DB, llm_func):
        self.action_list = None
        self.MAX_RETRIES = max_tries
        self.sensitive_actions = sensitive_actions
        self.replacements = replacements
        self.DB = DB
        self.semaphore = semaphore  
        self.call_llm_func = llm_func

    async def get_action_chunks(self):
        return await self.DB.get_all_shadow_entity_chunks()
    
    async def mark_sensitive_action_text_spans(self, chunk_id, chunk, action_dict):
        marked_chunk = chunk['content'].lower()
        filtered_action_dict = {}

        for action in action_dict:
            lowercased_action_span = action.lower()
            pattern = re.compile(rf"\b{re.escape(lowercased_action_span)}\b")

            if not pattern.search(marked_chunk):
                continue

            filtered_action_dict[lowercased_action_span] = action_dict[action]
            marked_chunk = pattern.sub(f"**<{lowercased_action_span}>**", marked_chunk)

        chunk['content'] = marked_chunk
        return filtered_action_dict, chunk

    async def swap_actions_in_chunk(self, chunk_id, chunk, action_dict):
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

    async def batch_actions(self, chunk):
        pass
    
    async def render_sensitive_actions(self, data):
    
        numbered_map = {
            i: {"label": key, "definition": value}
            for i, (key, value) in enumerate(data.items(), start=1)
        }

        formatted_string = "\n".join(
            f"{i}: {entry['label']} - {entry['definition']}"
            for i, entry in numbered_map.items()
        )

        return numbered_map, formatted_string

    async def extract_sensitive_actions_from_chunk(self, chunk, chunk_id):

        number_to_type, _ = await self.render_sensitive_actions(self.sensitive_actions)

        PROMPT = f"""
        <Role>
        You are an assistant tasked with extracting all text spans from the provided text chunk that meet the provided action classification types that should be replaced. You will extract exact text spans. You are not inferring or predicting actions not stated directly in the passage. You are not creating new sensitive actions classes.
        </Role>

       <Instructions>
        1. Read the sensitive action classes.
        2. Identify all exact text spans that contain an action from the text chunk that meet the definition of one of the provided sensitive action classifications. Extract as many as possible. Rarely, you may have zero sensitive actions.
        3. Your answer should be a JSON where the key is the exact text span and its value is the number of its associated classification.
        </Instructions>

        <Allowed Sensitive Entity Classes>
        There exist {len(self.sensitive_actions.keys())} possible sensitive actions. An action can only be of one class. Keep this in mind.
        {number_to_type}
        </Allowed Sensitive Entity Classes>

        <Text Chunk>
        {chunk}
        </Text Chunk>

        <Example Output Format>
        {{"< Write the exact sensitive text span from the chunk>": "write its associated numerical value type", ...}}
        </Example Output Format>
        """
        action_dict = json.loads(await self.call_llm_func(
        messages=[{"role": "user", "content": PROMPT}]
        ))

        print("This is what the llm sees", PROMPT)

        action_with_labels = {
            action: number_to_type[int(type_id)]["label"]
            for action, type_id in action_dict.items()
        }

        enriched_response = {}

        for action, desc in action_dict.items():
            label = action_with_labels.get(action, "Unknown")

            if label == "Unknown":
                print(f"Warning: Action '{action}' has an unknown label. Skipping enrichment.")
                continue

            # md5 hash
            action_hash = hashlib.md5(action.encode()).hexdigest()
            action_id = f"action-{action_hash}"

            enriched_response[action] = {
                "new_action": "",
                "chunk_id": chunk_id,
                "action_id": action_id,
                "choice": desc,
                "type": label
            }

        # LLM MAKE CHOICES

        await self.make_action_replacement_changes(enriched_response)
        return enriched_response

    async def make_action_replacement_changes(self, enriched_response):
        import json

        span_to_id = {}
        id_to_span = {}
        instructions = []

        # Step 1: assign ids and build instructions using replacements
        for idx, (span, data) in enumerate(enriched_response.items(), start=1):
            span_to_id[span] = idx
            id_to_span[idx] = span

            action_type = data["type"]
            replacement_instruction = self.replacements.get(action_type, "")

            instructions.append(
                f'{idx}: Rewrite the text span "{span}" by {replacement_instruction}. Leave blank if not possible.'
            )

        instruction_block = "\n".join(instructions)

        # sample output format
        sample_output = {
            idx: "<Write rewritten span here>"
            for idx in id_to_span
        }

        PROMPT = f"""
    <Role>
    You are editing text spans by making minimal token substitutions only.
    </Role>

    <Instructions>
    - You will be given numbered rewrite tasks.
    - You MUST NOT rewrite the sentence.
    - You MUST NOT change sentence structure, order, or grammar.
    - You MUST ONLY replace a FEW specific words.

    - Allowed:
        - swap verbs
        - swap small phrases
        - slight wording changes

    - NOT allowed:
        - rewriting the sentence
        - reordering words
        - adding new clauses
        - removing major parts of the sentence

    - The output MUST closely resemble the original text span.

    - CRITICAL RULE:
    You MUST NOT modify anything inside **[ ... ]**.
    These tokens must remain EXACTLY unchanged.

    - If you cannot make a minimal change, return an empty string "".

    </Instructions>

    <Rewrite Tasks>
    {instruction_block}
    </Rewrite Tasks>

    <Output Format>
    - Return ONLY valid JSON.
    - Keys must be the task numbers.
    - Values must be the rewritten spans.

    {json.dumps(sample_output, indent=2)}
    </Output Format>
    """

        print("prompt", PROMPT)

        response = json.loads(await self.call_llm_func(
            messages=[{"role": "user", "content": PROMPT}]
        ))

        print("choices", json.dumps(response, indent=4))

        return response, span_to_id, id_to_span