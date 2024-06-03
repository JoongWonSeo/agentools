from typing import Literal

from groq import AsyncGroq
from openai import AsyncOpenAI


class Model:
    class Mock: ...

    class Echo: ...

    class Replay: ...

    GroqClient = AsyncGroq
    OpenAIClient = AsyncOpenAI

    def __init__(
        self,
        client: Literal["mock", "echo", "groq", "openai", "local", "replay", None],
        model_name: str | None = None,
        url: str | None = None,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            client: The client to use. Can be one of "mock", "echo", "replay", "groq", "local", or "openai".
            model_name: The name of the model. If client=mock, this will be the thing returned.
            url: The URL to use for the "local" client option. Optional.

        Raises:
            ValueError: If an invalid client option is provided.

        """
        _client = None

        if client == "mock":
            _client = self.Mock()
            if not model_name:
                model_name = "Hello, world!"
        elif client == "echo":
            _client = self.Echo()

        elif client == "replay":
            _client = self.Replay()

        elif client == "groq":
            _client = self.GroqClient()
        elif client == "local":
            _client = self.OpenAIClient(base_url=url)
        elif client == "openai":
            _client = self.OpenAIClient()
        else:
            raise ValueError("Invalid client option provided.")

        self.client = _client
        self.model_name = model_name

    @classmethod
    def default(cls):
        """The default model is OpenAI's GPT-3.5 Turbo."""
        return Model(client="openai", model_name="gpt-3.5-turbo", url=None)
