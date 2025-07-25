import base64
import re
from typing import Iterable, Literal, overload

from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam as ImageContent,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import (
    ChatCompletionContentPartInputAudioParam as InputAudioContent,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.chat.chat_completion_content_part_param import (
    ChatCompletionContentPartParam as Content,
)
from openai.types.chat.chat_completion_content_part_refusal_param import (
    ChatCompletionContentPartRefusalParam as RefusalContent,
)
from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam as TextContent,
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
def content(*, image_url: str | ImageURL | None) -> ImageContent:
    """
    OpenAI image content part: {"type": "image_url", "image_url": ...}
    """
    ...


type AudioData = str | bytes | bytearray | memoryview


@overload
def content(
    *, input_audio: AudioData, format: Literal["wav", "mp3"]
) -> InputAudioContent:
    """OpenAI audio content part: {"type": "input_audio", "input_audio": ...}"""
    ...


@overload
def content(*, input_audio: InputAudio) -> InputAudioContent:
    """OpenAI audio content part: {"type": "input_audio", "input_audio": ...}"""
    ...


def content(
    *,
    text=None,
    image_url: str | ImageURL | None = None,
    detail: Literal["auto", "low", "high"] | None = None,
    input_audio: AudioData | InputAudio | None = None,
    format: Literal["wav", "mp3"] | None = None,
) -> Content:
    # text content
    if text is not None:
        assert all(arg is None for arg in (image_url, detail, input_audio, format)), (
            "Text content should not have other arguments"
        )
        return {"type": "text", "text": text}

    # image content
    elif image_url is not None:
        assert all(arg is None for arg in (text, input_audio, format)), (
            "Image content should not have other arguments"
        )
        if isinstance(image_url, dict):
            # ImageURL dict
            assert detail is None, "ImageURL should not have detail argument"
            # return {"type": "image_url", "image_url": image_url}
            return ImageContent(type="image_url", image_url=image_url)
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
        assert all(arg is None for arg in (text, image_url, detail)), (
            "Audio content should not have other arguments"
        )
        if isinstance(input_audio, dict):
            # InputAudio dict
            assert format is None, "InputAudio should not have format argument"
            return {"type": "input_audio", "input_audio": input_audio}
        else:
            assert isinstance(input_audio, (str, bytes, bytearray, memoryview)), (
                "input_audio should be str or byte-like"
            )
            assert format is not None, "Audio format should be provided"
            if not isinstance(input_audio, str):
                input_audio = base64.b64encode(input_audio).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": InputAudio(data=input_audio, format=format),
            }

    else:
        raise ValueError("One of text/image/audio should be specified")


def ensure_content(content: Iterable[Content | RefusalContent]) -> list[Content]:
    """
    Raises ValueError if any content is a refusal.
    """
    if any(c["type"] == "refusal" for c in content):
        raise ValueError("Refusal content is not allowed")
    return [c for c in content if c["type"] != "refusal"]


def format_contents(
    prompt_template: str,
    _partial: bool = False,
    _strip: bool = True,
    **substitutions: str | list[Content],
) -> list[Content]:
    """
    Returns a list of content items by substituting placeholders in `prompt_template`.

    - Single braces {key} are replaced with either:
        * a string (appended to the current text buffer),
        * a list of Content (causes the current text buffer to flush, then these items are inserted),
        * or if the key is missing:
            - raise ValueError unless _partial=True, in which case {key} is left as literal text.
    - Use double braces {{...}} to escape a placeholder.
    - The result is a list of Content dicts.
      Contiguous text (including any single‐brace string substitutions and double‐brace literals)
      is combined into one "text" block until a list substitution is encountered or we reach the end.

    Example:
    ```
    format_contents(
        "Hello, {name}, see {{not a placeholder}}: {image}!",
        name="Alice",
        image=[{'type': 'image_url', 'image_url': {'url': 'https://example.com/image.jpg'}}]
    )
    -> [
        {'type': 'text',
            'text': 'Hello, Alice, see {not a placeholder}:'},
        {'type': 'image_url',
            'image_url': {'url': 'https://example.com/image.jpg'}},
        {'type': 'text', 'text': '!'}
    ]
    ```
    """

    # Regex to match either:
    #   1) {{...}} (double braces) as literal text
    #   2) {some_key} (single braces) as a placeholder
    pattern = re.compile(r"(\{\{[^}]*\}\}|\{\w+\})")

    result: list[Content] = []
    text_buffer: list[str] = []  # We'll accumulate text here and flush as one chunk.

    last_index = 0
    for match in pattern.finditer(prompt_template):
        start, end = match.span()
        matched_text = match.group(0)

        # Text before the match goes to the buffer
        if start > last_index:
            text_buffer.append(prompt_template[last_index:start])

        if matched_text.startswith("{{") and matched_text.endswith("}}"):
            # Double-braced content -> literal text
            text_buffer.append(matched_text.replace("{{", "{").replace("}}", "}"))
        else:
            # Single-braced placeholder: {key}
            placeholder_name = matched_text[1:-1]  # strip braces

            # Check if missing
            if placeholder_name not in substitutions:
                if _partial:
                    # Keep the placeholder as literal text
                    text_buffer.append(matched_text)
                else:
                    raise ValueError(
                        f"Missing substitution for placeholder '{placeholder_name}'"
                    )
            else:
                substitution = substitutions[placeholder_name]
                if isinstance(substitution, str):
                    # Append string substitution directly into the text buffer
                    text_buffer.append(substitution)
                elif isinstance(substitution, list):
                    # Flush text buffer as a single chunk (if non-empty)
                    new_text = "".join(text_buffer)
                    next_text = new_text.strip() if _strip else new_text
                    if next_text:
                        result.append({"type": "text", "text": next_text})
                        text_buffer = []
                    # Extend result by the content list
                    result.extend(substitution)
                else:
                    # Fallback: treat any other type as string
                    text_buffer.append(str(substitution))

        last_index = end

    # Add any remaining text after the last match
    if last_index < len(prompt_template):
        text_buffer.append(prompt_template[last_index:])

    # Flush any final accumulated text
    new_text = "".join(text_buffer)
    next_text = new_text.strip() if _strip else new_text
    if next_text:
        result.append({"type": "text", "text": next_text})

    return result
