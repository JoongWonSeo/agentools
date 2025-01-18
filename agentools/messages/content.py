import base64
from typing import Literal, overload


from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam as Content,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as TextContent,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam as ImageContent,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam as InputAudioContent,
    InputAudio,
)


@overload
def content(*, text: str) -> TextContent:
    """OpenAI text content part: {"type": "text", "text": ...}"""
    ...


@overload
def content(
    *,
    image_url: str,
    detail: Literal["auto", "low", "high"] | None = None,
    # format: Literal["jpeg", "jpg", "png", "gif", "webp"],
) -> ImageContent:
    """
    OpenAI image content part: {"type": "image_url", "image_url": ...}

    Args:
        image_url: either the url or base64-encoded image
        detail: image detail level (auto, low, high), None is auto
    """
    ...


@overload
def content(*, image_url: ImageURL) -> ImageContent:
    """
    OpenAI image content part: {"type": "image_url", "image_url": ...}
    """
    ...


@overload
def content(
    *, input_audio: str | bytes, format: Literal["wav", "mp3"]
) -> InputAudioContent:
    """OpenAI audio content part: {"type": "input_audio", "input_audio": ...}"""
    ...


@overload
def content(*, input_audio: InputAudio) -> InputAudioContent:
    """OpenAI audio content part: {"type": "input_audio", "input_audio": ...}"""
    ...


def content(
    *, text=None, image_url=None, detail=None, input_audio=None, format=None
) -> Content:
    # text content
    if text is not None:
        assert all(
            arg is None for arg in (image_url, detail, input_audio, format)
        ), "Text content should not have other arguments"
        return {"type": "text", "text": text}

    # image content
    elif image_url is not None:
        assert all(
            arg is None for arg in (text, input_audio, format)
        ), "Image content should not have other arguments"
        if isinstance(image_url, dict):
            # ImageURL dict
            assert detail is None, "ImageURL should not have detail argument"
            return {"type": "image_url", "image_url": image_url}
        else:
            assert isinstance(image_url, str), "image_url should be str"
            # image url
            # TODO: what happens if detail=None?
            return {
                "type": "image_url",
                "image_url": ImageURL(url=image_url, detail=detail)
                if detail
                else ImageURL(url=image_url),
            }

    # audio content
    elif input_audio is not None:
        assert all(
            arg is None for arg in (text, image_url, detail)
        ), "Audio content should not have other arguments"
        if isinstance(input_audio, dict):
            # InputAudio dict
            assert format is None, "InputAudio should not have format argument"
            return {"type": "input_audio", "input_audio": input_audio}
        else:
            assert isinstance(
                input_audio, (str, bytes)
            ), "input_audio should be str or bytes"
            assert format is not None, "Audio format should be provided"
            if isinstance(input_audio, bytes):
                input_audio = base64.b64encode(input_audio).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": InputAudio(data=input_audio, format=format),
            }

    else:
        raise ValueError("One of text/image/audio should be specified")
