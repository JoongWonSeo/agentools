from __future__ import annotations
from typing import Type, ClassVar

from pydantic import BaseModel, Field

from .embedding import EmbeddingModel, OpenAIEmbedding


# TODO: field to indicate indexing
# def IndexField(...)


def EmbeddableField(embedder: Type[EmbeddingModel] = OpenAIEmbedding, *args, **kwargs):
    """
    A field that can be embedded into a vector database. Currently only supports str.
    You can specify the embedding model to use, or use the default OpenAIEmbedding.
    """
    json_schema_extra = kwargs.pop("json_schema_extra", {})

    if embedder:
        EmbeddableData.embedder_registry[embedder.__name__] = embedder
        json_schema_extra["embed"] = True
        json_schema_extra["embedding_model"] = embedder.__name__
        json_schema_extra["embedding_dim"] = embedder.EMBEDDING_DIM
        json_schema_extra["embedding_distance"] = embedder.DISTANCE

    return Field(*args, **kwargs, json_schema_extra=json_schema_extra)


class EmbeddableData(BaseModel):
    """
    Abstract class for a data model that contains at least one embeddable field.
    Mark the embeddable field with ... = EmbeddableField(EmbeddingModel).
    """

    embedder_registry: ClassVar[
        dict[str, Type[EmbeddingModel]]
    ] = {}  # (global) embedding cls name -> cls
    """A dictionary of Embedding Models that you can access globally.

    Key:
        EmbeddingModel's Class name.
    Value:
        EmbeddingModel's Class

    Example: EmbeddableData.embedder_registry
    """

    def to_payload(self):
        """Return a dict that can be stored in the vector database."""
        return self.model_dump()

    @classmethod
    def field_embedder(cls):
        """Return a dict of EmbeddableField names and their EmbeddingModel's classes."""

        return {
            name: EmbeddableData.embedder_registry[
                field.json_schema_extra["embedding_model"]
            ]
            for name, field in cls.model_fields.items()
            if field.json_schema_extra and field.json_schema_extra.get("embed")
        }

    @classmethod
    async def embed(
        cls, batch: list[EmbeddableData], field_embedder=None, batch_size=None
    ) -> list[dict[str, list[float]]]:
        """
        Given a batch of EmbeddableData, create a batch of embeddings for each vector field.
        All EmbeddableData in the batch must be of the same type.
        """
        assert all(
            isinstance(d, cls) for d in batch
        ), f"Batch must be of type {cls.__name__}, got a mix of { {b.__class__.__name__ for b in batch} }"
        field_embedder = field_embedder or cls.field_embedder()

        # gather by field, and batch-embed each field
        field_batches = {
            field_name: await embedding_model_cls.embed(
                [getattr(d, field_name) for d in batch]
            )
            for field_name, embedding_model_cls in field_embedder.items()
        }
        # transpose batch
        return [
            {
                field_name: field_batch[i]
                for field_name, field_batch in field_batches.items()
            }
            for i in range(len(batch))
        ]
