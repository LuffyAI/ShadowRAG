import aiohttp
from .EmbeddingModel import OpenRouterEmbedding

class OpenRouterChat:
    def __init__(self, api_key, model="meta-llama/llama-3.1-8b-instruct", embedding_model="text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self._session = None

        # attach embedding model as attribute
        self.embedding = OpenRouterEmbedding(api_key, embedding_model)

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate(self, messages, temperature=0):
        session = await self._get_session()

        async with session.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
                "reasoning": {"exclude": True},
                "messages": messages,
            },
        ) as res:
            data = await res.json()
            print("Full response from OpenRouter Chat API:", data)  # Debugging line
            return data["choices"][0]["message"]["content"]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        await self.embedding.close()