from agentools.messages import format_contents, content


def test_format_contents_mix():
    result = format_contents(
        "Hello {name}! {img}", name="Bob", img=[content(text="image")]
    )
    assert result[0]["type"] == "text"
    assert result[1]["type"] == "text"
