import json
import hashlib
import asyncio
import re
from difflib import ndiff


from torch import chunk

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
    
    async def check_token_diff(self, rebuilt_sentence, sentence):
            
            print("Rebuilt Sentence:", rebuilt_sentence)
            diff = list(ndiff(sentence.split(), rebuilt_sentence.split()))

            changes = {
                "added": [d[2:] for d in diff if d.startswith("+ ")],
                "removed": [d[2:] for d in diff if d.startswith("- ")]
            }

            print("Diff:", json.dumps(changes, indent=4))
    
    async def identify_and_replace_sensitive_tokens_in_sentence(self, chunk_id, sentence):
        import json

        # Step 1: tokenize (simple whitespace split)
        tokens = sentence.split()

        token_dict = {
            idx: token
            for idx, token in enumerate(tokens)
        }

        # expected output format (token → replacement or "")
        expected_output = {
            token: "<Write replacement or \"\" if unchanged>"
            for token in tokens
        }

        PROMPT = f"""
        <Role>
        You are identifying and minimally replacing sensitive tokens in a sentence.
        </Role>

        <Instructions>
        1. Read the sensitive action classes and their replacement strategies.
        2. You are given a tokenized sentence.
        3. ONLY replace tokens that directly express a sensitive action.
        4. All other tokens MUST be left unchanged and assigned "".
        5. You MUST NOT rewrite the sentence.
        6. You MUST NOT change token order.
        7. You MUST NOT merge or split tokens.

        8. CRITICAL RULE:
        You MUST NOT modify anything inside **[ ... ]** tokens.
        These must ALWAYS return "".

        9. IMPORTANT BEHAVIOR RULE:
        - The replacement rules are GUIDELINES, NOT output text.
        - You MUST NOT copy or repeat the replacement rule text.
        - You MUST APPLY the rule to generate a SHORT, natural replacement token.
        - The replacement must fit grammatically into the sentence.
        - The replacement should be concise (1–3 words max).

        10. MATCHING RULE:
        - When matching tokens, IGNORE punctuation.
        - Example: "companies," should match "companies".

        11. Output a JSON where:
        - key = original token (exactly as given)
        - value = replacement token OR "" if unchanged

        </Instructions>

        <Sensitive Actions and Replacement Rules>
        {json.dumps(self.replacements, indent=2)}
        </Sensitive Actions and Replacement Rules>

        <Tokens>
        {json.dumps(token_dict, indent=2)}
        </Tokens>

        <Output Format (MUST MATCH EXACTLY)>
        {json.dumps(expected_output, indent=2)}
        </Output Format>
        """

        response = json.loads(await self.call_llm_func(
            messages=[{"role": "user", "content": PROMPT}]
        ))

        print("Token Replacement Prompt:", PROMPT)
        print("Token Replacement Output:", json.dumps(response, indent=4))

        # rebuild sentence
        new_tokens = []
        for token in tokens:
                replacement = response.get(token, "")
                new_tokens.append(replacement if replacement != "" else token)

        # fix spacing
        rebuilt_sentence = ""
        for i, tok in enumerate(new_tokens):
                if i == 0:
                    rebuilt_sentence += tok
                elif tok in [".", ",", "!", "?", ":", ";"]:
                    rebuilt_sentence += tok
                else:
                    rebuilt_sentence += " " + tok

        await self.DB.insert_action_chunk_pair(chunk_id, sentence, rebuilt_sentence)

        return rebuilt_sentence
        
    async def break_chunk_into_sentences(self, chunk):

        # split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())

        # label as 1a, 2a, 3a...
        sentence_dict = {
            f"{idx}a": sentence
            for idx, sentence in enumerate(sentences, start=1)
            if sentence  # skip empty
        }

        return sentence_dict
    
    async def shadow_action(self, chunk, chunk_id):

            PROMPT = f"""
            <Role>
            Carefully read the text chunk and apply the sensitive action modifications.
            </Role>

            <Instructions>
            Switch all sensitive terms with the suggested rules.

            Rules:
            - Extract sensitive text spans
            - Replace them so it transforms to the suggested context
            - Do NOT modify unrelated text

            Return ONLY JSON in this format:
            {{"original": "replacement"}}
            </Instructions>

            <Allowed Sensitive Action Classes>
            There exist {len(self.sensitive_actions.keys())} possible sensitive actions.
            {await self.render_sensitive_actions(self.sensitive_actions)}
            </Allowed Sensitive Action Classes>

            <Text Chunk>
            {chunk}
            </Text Chunk>
            """

            classification_dict = json.loads(await self.call_llm_func(
                messages=[{"role": "user", "content": PROMPT}]
            ))

            print("Classification Prompt:", PROMPT)
            print("Classification Output:", json.dumps(classification_dict, indent=4))

            text = await self.apply_shadow_changes(chunk, chunk_id, classification_dict)


            return text
    
    async def apply_shadow_changes(self, chunk, chunk_id, classification_dict):
        import re

        text = chunk

        original = classification_dict["original"]
        replacement = classification_dict["replacement"]

        # protect **[ ... ]**
        protected_spans = list(re.finditer(r"\*\*\[[^\]]+\]\*\*", text))

        def is_inside_protected(start, end):
            for m in protected_spans:
                if start >= m.start() and end <= m.end():
                    return True
            return False

        for match in list(re.finditer(re.escape(original), text)):
            start, end = match.start(), match.end()

            if is_inside_protected(start, end):
                continue

            text = text[:start] + f"<{replacement}>" + text[end:]
            break  # only replace once

        await self.DB.insert_shadow_chunk(chunk_id, text)

        return text

    
    # async def classify_chunk_sentences(self, chunk, chunk_id):
    #     import json

    #     sentence_dict = await self.break_chunk_into_sentences(chunk)

    #     # 🔥 build exact expected output
    #     expected_output = {
    #         key: "<Insert either 0 for no or 1 for yes>"
    #         for key in sentence_dict.keys()
    #     }

    #     PROMPT = f"""
    # <Role>
    # You are performing a strict binary classification task.
    # </Role>

    # <Instructions>
    # 1. Read the sensitive action classes.
    # 2. For each sentence, determine if it explicitly contains a sensitive action.
    # 3. You MUST NOT infer anything not directly stated.
    # 4. You MUST assign:
    # - 1 → contains a sensitive action
    # - 0 → does NOT contain a sensitive action

    # 5. You MUST return EXACTLY the same JSON structure provided below.
    # 6. Do NOT add or remove keys.
    # 7. Do NOT change key names.
    # </Instructions>

    # <Allowed Sensitive Action Classes>
    # There exist {len(self.sensitive_actions.keys())} possible sensitive actions.
    # {await self.render_sensitive_actions(self.sensitive_actions)}
    # </Allowed Sensitive Action Classes>

    # <Sentences to Classify>
    # {json.dumps(sentence_dict, indent=2)}
    # </Sentences to Classify>

    # <Output Format (MUST MATCH EXACTLY)>
    # {json.dumps(expected_output, indent=2)}
    # </Output Format>
    # """

    #     classification_dict = json.loads(await self.call_llm_func(
    #         messages=[{"role": "user", "content": PROMPT}]
    #     ))

    #     print("Classification Prompt:", PROMPT)
    #     print("Classification Output:", json.dumps(classification_dict, indent=4))

    #     await self.identify_and_replace_sensitive_tokens_in_sentence(chunk_id, sentence_dict['1a'])

    #     return classification_dict





    async def extract_sensitive_actions_from_chunk(self, chunk, chunk_id):

        print("Extracting sensitive actions from chunk:", await self.break_chunk_into_sentences(chunk))
        raise("stop")
        
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