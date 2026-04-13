import os
import tiktoken
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv() 


class Tokenizer:
    def __init__(self, model_name: str):
        self.model_name = model_name

        # Decide which tokenizer to use
        if self._is_openai_model(model_name):
            self.backend = "tiktoken"
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            self.backend = "hf"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))

    def _is_openai_model(self, model_name: str) -> bool:
        # Simple heuristic — extend as needed
        return any(x in model_name.lower() for x in [
            "gpt", "o1", "o3", "o4", "text-embedding"
        ])

    def encode(self, text: str) -> list[int]:
        if self.backend == "tiktoken":
            return self.tokenizer.encode(text)
        else:
            return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        if self.backend == "tiktoken":
            return self.tokenizer.decode(tokens)
        else:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))