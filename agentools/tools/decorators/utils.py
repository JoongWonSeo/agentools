import inspect
import json
from typing import Callable
from typing_extensions import deprecated

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model, ValidationError as PydanticValError
from jsonschema import Draft202012Validator
import jsonref


class ValidationError(BaseException):
    pass


# Function -> Pydantic model
def pydantic_from_doc(
    func: Callable,
    name: str | None = None,
    require_doc=True,
    use_only_description=False,
) -> type[BaseModel]:
    """
    Convert a function to a pydantic model for validation of function parameters.

    Args:
        func: function to convert
        name: name of the model, defaults to func.__name__
        require_doc: whether to require a docstring to be present for the function
        use_only_description: whether to use only the overall description from the docstring, instead of each parameter's description

    Returns:
        pydantic model, validator function
    """
    sig: inspect.Signature = inspect.signature(func)
    doc = parse(inspect.getdoc(func) or "")

    # get function name
    name = name or func.__name__

    # get general function description
    summary = f"{doc.short_description or ''}\n{doc.long_description or ''}".strip()

    # {param_name: param_description}
    param_descriptions = {
        doc_param.arg_name: doc_param.description for doc_param in doc.params
    }

    # convert function parameters to pydantic model fields
    model_fields = {}

    for param in sig.parameters.values():
        if param.name in ["self", "return", "return_type"]:
            continue

        if require_doc and not use_only_description:
            assert (
                param.name in param_descriptions
            ), f"Missing description for parameter `{param.name}` in `{func.__name__}`'s docstring!"

        assert (
            param.annotation != inspect.Parameter.empty
        ), f"Missing type annotation for parameter `{param.name}` in `{func.__name__}`!"

        model_fields[param.name] = (
            param.annotation,  # type
            Field(
                title=None,
                description=param_descriptions.get(param.name),
                default=...
                if param.default is inspect.Parameter.empty
                else param.default,
            ),
        )

    # create a pydantic model for function argument validation
    model = create_model(name, __doc__=summary or None, **model_fields)

    return model


# JSON Schema -> JSON Validator that raises ValidatonError just like Pydantic Model
def validator_from_schema(
    json_schema: dict,
    name: str | None = None,
) -> Callable:
    if name:
        json_schema["title"] = name

    # Validate json against schema
    def validate_json(**state):
        v = Draft202012Validator(json_schema)
        if errors := "\n".join({e.message for e in v.iter_errors(state)}):
            raise ValidationError(errors)
        return True

    return validate_json


# Pydantic model -> Pydantic validator function
@deprecated("First convert pydantic to JSONSchema and use validator_from_schema")
def validator_from_pydantic(model: type[BaseModel]) -> Callable:
    # validator function
    def validate_pydantic(**state):
        try:
            model(**state)
        except PydanticValError as e:
            err = "\n".join(
                [
                    line
                    for line in str(e).splitlines()
                    if "further information" not in line.lower()
                ]
            )
            raise ValidationError(err)

    return validate_pydantic


# Set JSON schema description from function docstring
def set_description(schema: dict, func: Callable, override: bool = False):
    """
    Set the description of the schema from the function docstring.

    Args:
        schema: JSON schema to set description for
        func: function to get docstring from
        override: whether to override the existing description, if False, empty docstrings will not override existing descriptions
    """

    doc = parse(inspect.getdoc(func) or "")
    parsed_description = (
        f"{doc.short_description or ''}\n{doc.long_description or ''}".strip()
    )
    if override or parsed_description:
        schema["description"] = parsed_description


# Pydantic/JSON Schema -> OpenAI function schema
def schema_to_openai_func(schema: dict | type[BaseModel], nested=True) -> dict:
    if not isinstance(schema, dict) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    if nested:
        schema = to_nested_schema(schema, no_title=False)

    # Convert properties
    remove_title(schema["properties"])

    # Construct the OpenAI function schema format
    return {
        "type": "function",
        "function": {
            "name": schema["title"],
            **({"description": d} if (d := schema.get("description")) else {}),
            "parameters": {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", []),
            },
        },
    }


def to_nested_schema(schema: dict, no_title=True) -> dict:
    """nested json schema rather than refs"""
    schema = jsonref.loads(json.dumps(schema), proxies=False)
    if "definitions" in schema:
        schema.pop("definitions")
    if "$defs" in schema:
        schema.pop("$defs")
    if no_title:
        remove_title(schema)
    return schema


def remove_title(d) -> dict | list:
    if isinstance(d, dict):
        if "title" in d and isinstance(d["title"], str):
            d.pop("title")
        for v in d.values():
            remove_title(v)
    elif isinstance(d, list):
        for v in d:
            remove_title(v)
    return d


# ========== Misc ========== #
async def awaitable(val):
    return val
