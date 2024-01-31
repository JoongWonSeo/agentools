from openai.types.completion_usage import CompletionUsage
from openai.types.chat import ChatCompletion


class ChatCompletionUsageTracker:
    MODEL_PRICING = {  # in $ / 1K tokens
        "gpt-4-1106-preview": {
            "prompt_tokens": 0.01,
            "completion_tokens": 0.03,
        },
        "gpt-4": {
            "prompt_tokens": 0.03,
            "completion_tokens": 0.06,
        },
        "gpt-4-32k": {
            "prompt_tokens": 0.06,
            "completion_tokens": 0.12,
        },
        "gpt-3.5-turbo-1106": {
            "prompt_tokens": 0.001,
            "completion_tokens": 0.002,
        },
        "gpt-3.5-turbo": {
            "prompt_tokens": 0.001,
            "completion_tokens": 0.002,
        },
        "gpt-3.5-turbo-0613": {
            "prompt_tokens": 0.001,
            "completion_tokens": 0.002,
        },
    }

    def __init__(self):
        self.usage_per_model: dict[str, CompletionUsage] = {}  # model -> usage

    def track_usage(self, completion: ChatCompletion) -> ChatCompletion:
        """Transparent wrapper to track usage of the completion"""
        model = completion.model
        usage = completion.usage
        if usage is None:
            return completion
        if model not in self.usage_per_model:
            self.usage_per_model[model] = CompletionUsage(
                completion_tokens=0, prompt_tokens=0, total_tokens=0
            )

        self.usage_per_model[model].completion_tokens += usage.completion_tokens
        self.usage_per_model[model].prompt_tokens += usage.prompt_tokens
        self.usage_per_model[model].total_tokens += usage.total_tokens

        print(
            f"Cost for {model}: ${self.calculate_cost(model, usage):0.2f}, Total: ${self.calculate_cost(model, self.usage_per_model[model]):0.2f}"
        )

        return completion

    @classmethod
    def calculate_cost(cls, model: str, usage: CompletionUsage) -> float:
        """Calculate cost of the completion"""
        if model not in cls.MODEL_PRICING:
            print(f"UsageTracker: Unknown Model {model}")
            return 0

        return (
            usage.completion_tokens
            / 1000
            * cls.MODEL_PRICING[model]["completion_tokens"]
            + usage.prompt_tokens / 1000 * cls.MODEL_PRICING[model]["prompt_tokens"]
        )
