import pytest


from agentools.retrieval import EmbeddableData, EmbeddableField, MockEmbedding


class Item(EmbeddableData):
    text: str = EmbeddableField(MockEmbedding)
    other: str


@pytest.mark.asyncio
async def test_embeddable_data_embed():
    data = [Item(text="a", other="b")]
    res = await Item.embed(data)
    assert isinstance(res[0]["text"], list)
    assert len(res[0]["text"]) == MockEmbedding.EMBEDDING_DIM
