import inspect
import json
from typing import Callable, Optional, Type

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model, ValidationError as PydanticValError
from jsonschema import Draft202012Validator
import jsonref


class ValidationError(BaseException):
    pass


# Function -> Pydantic model
def validator_from_doc(
    func: Callable, name: Optional[str] = None, require_doc=True
) -> type[BaseModel]:
    """
    Convert a function to a pydantic model for validation of function parameters.

    Args:
        func: function to convert
        name: name of the model, defaults to func.__name__

    Returns:
        pydantic model
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

        if require_doc:
            assert (
                param.name in param_descriptions
            ), f"Missing description for parameter {param.name} in {func.__name__}'s docstring!"

        assert (
            param.annotation != inspect.Parameter.empty
        ), f"Missing type annotation for parameter {param.name} in {func.__name__}!"

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

    # validator function
    def validate_pydantic(**state):
        try:
            model(**state)
        except PydanticValError as e:
            err = "\n".join(
                [
                    l
                    for l in str(e).splitlines()
                    if "further information" not in l.lower()
                ]
            )
            raise ValidationError(err)

    return model, validate_pydantic


# JSON Schema -> JSON Validator that raises ValidatonError just like Pydantic Model
def validator_from_schema(
    json_schema: dict,
    name: Optional[str] = None,
    override_with_doc_from: Optional[Callable] = None,
) -> Callable:
    if name:
        json_schema["title"] = name
    if override_with_doc_from:
        doc = parse(inspect.getdoc(override_with_doc_from) or "")
        json_schema[
            "description"
        ] = f"{doc.short_description or ''}\n{doc.long_description or ''}".strip()

    # Validate json against schema
    def validate_json(**state):
        v = Draft202012Validator(json_schema)
        if errors := "\n".join({e.message for e in v.iter_errors(state)}):
            raise ValidationError(errors)
        return True

    return validate_json


# Pydantic/JSON Schema -> OpenAI function schema
def schema_to_openai_func(schema: dict | Type[BaseModel], nested=True) -> dict:
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
        if "title" in d and type(d["title"]) == str:
            d.pop("title")
        for v in d.values():
            remove_title(v)
    elif isinstance(d, list):
        for v in d:
            remove_title(v)
    return d
