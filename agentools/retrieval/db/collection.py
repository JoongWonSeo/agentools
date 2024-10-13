from __future__ import annotations
from typing import Generic, TypeVar
from uuid import uuid4
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    VectorParams,
    PointStruct,
    Filter,
)

from .data import EmbeddableData


D = TypeVar("D", bound=EmbeddableData)


class EmbeddableDataCollection(Generic[D]):
    """
    A vector database that represents a collection of a specific EmbeddableData model.
    """

    global_client = None

    @staticmethod
    def use_global_client(client: AsyncQdrantClient = None):
        """Set a global client that is used as default for all VectorDatabaseClient instances, mostly for convenience and testing."""
        EmbeddableDataCollection.global_client = client or AsyncQdrantClient(
            path=Path("storage", "qdrant")
        )

    def __init__(self, name: str, data_model: type[D], client=None, validate=True):
        """Instantiate a collection. If it doesn't exist, it creates a new one."""

        self.name = name
        self.data_model = data_model
        self.field_embedder = data_model.field_embedder()
        self.client = client or EmbeddableDataCollection.global_client

        assert self.client, "No client provided and no global client set."

        if validate:
            self._validate_vector_fields()

    async def check_exist_and_initialize(self) -> bool:
        """Use existing collection or create if necessary, and return whether it exists."""
        does_exist = self.name in [
            c.name for c in (await self.client.get_collections()).collections
        ]

        if not does_exist:
            await self.reset()

        return does_exist

    async def reset(self) -> bool:
        """Reset the collection, or create a new one if it doesn't exist yet."""
        operation_result = await self.client.recreate_collection(
            self.name,
            {
                field: VectorParams(size=embed.EMBEDDING_DIM, distance=embed.DISTANCE)
                for field, embed in self.field_embedder.items()
            },
        )

        return operation_result

    async def destroy(self):
        await self.client.delete_collection(self.name)

    async def add(self, data: list[D], ids: list[str | int] = None):
        if not ids:
            ids = [str(uuid4()) for _ in range(len(data))]

        assert len(data) == len(ids)

        embeddings = await self.data_model.embed(data, self.field_embedder)

        points = [
            PointStruct(id=i, vector=e, payload=d.to_payload())
            for i, d, e in zip(ids, data, embeddings)
        ]

        operation_info = await self.client.upsert(
            collection_name=self.name, points=points, wait=True
        )
        assert operation_info.status == "completed"

    async def overwrite(self, data: D, id: str | int):
        operation_info = await self.client.overwrite_payload(
            collection_name=self.name, points=[id], payload=data.to_payload(), wait=True
        )
        assert operation_info.status == "completed"

    async def query(
        self, top: int = 10, filter: Filter = None, **query_args
    ) -> list[tuple[float, D]]:
        """
        Query using a keyword argument of desired field name of EmbeddableData and its value.
        Returns a list of (score, EmbeddableData) tuples.
        """
        assert len(query_args) == 1
        field_name, query = list(query_args.items())[0]
        embedder = self.field_embedder[field_name]

        query_vector = (await embedder.embed([query]))[0]
        search_result = await self.client.search(
            collection_name=self.name,
            query_vector=(field_name, query_vector),
            limit=top,
            query_filter=filter,
        )
        return [(r.score, self.data_model(**r.payload)) for r in search_result]

    async def retrieve(self, ids: list[str | int]) -> list[D]:
        """
        Retrieve EmbeddableData by id.
        """
        points = await self.client.retrieve(
            self.name, ids=ids, with_payload=True, with_vectors=False
        )
        return [self.data_model(**p.payload) for p in points]

    async def iterate(
        self, filter: Filter = None, batch: int = 100, with_id: bool = False
    ):
        """
        Iterate over all documents in the collection.
        """
        offset = None
        while True:
            points, offset = await self.client.scroll(
                self.name,
                scroll_filter=filter,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                model = self.data_model(**point.payload)
                yield (point.id, model) if with_id else model

            if offset is None:
                break

    def __aiter__(self):
        return self.iterate()

    async def get(self, key: str | int, default=None):
        return self.data_model(
            **(await self.client.retrieve(self.name, ids=[key]))[0].payload
        )

    async def size(self):
        return (await self.client.get_collection(self.name)).points_count

    def _validate_vector_fields(self):
        if not self.field_embedder:
            raise ValueError(
                "The model must have at least one field marked as vectorized."
            )
        for field_name in self.field_embedder:
            field = self.data_model.model_fields[field_name]
            if field.annotation is not str:
                raise TypeError(f"Vectorized field {field_name} must be of type str.")
