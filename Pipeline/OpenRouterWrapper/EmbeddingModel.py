import aiohttp

class OpenRouterEmbedding:
    def __init__(self, api_key, model="baai/bge-large-en-v1.5"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/embeddings"
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def embed(self, text):
        session = await self._get_session()
        async with session.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text,
            },
        ) as res:
            data = await res.json()
            return data["data"][0]["embedding"]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()