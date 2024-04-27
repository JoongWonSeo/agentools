from abc import ABC

from openai import AsyncOpenAI
from qdrant_client.http.models import Distance


class EmbeddingModel(ABC):
    """Abstract Base Class for all text embedding models"""

    EMBEDDING_DIM: int = -1
    DISTANCE: Distance = Distance.COSINE

    @classmethod
    async def embed(cls, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)


class MockEmbedding(EmbeddingModel):
    EMBEDDING_DIM = 2
    DISTANCE = Distance.COSINE

    @classmethod
    async def embed(cls, texts: list[str]) -> list[list[float]]:
        return [[1.0] * cls.EMBEDDING_DIM] * len(texts)


class OpenAIEmbedding(EmbeddingModel):
    EMBEDDING_DIM = 1536
    DISTANCE = Distance.COSINE

    @classmethod
    async def embed(cls, texts: list[str]) -> list[list[float]]:
        EMBEDDING_MODEL = "text-embedding-3-small"
        client = AsyncOpenAI()

        # TODO: there is an error if an element in [texts] has > 8192 tokens

        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [r.embedding for r in response.data]
