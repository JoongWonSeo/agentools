from abc import ABC, abstractmethod
from typing import Type, Optional, Callable
import inspect
import json
import asyncio
from functools import wraps
from itertools import chain

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model, ValidationError as PydanticValError
from jsonschema import Draft202012Validator
import jsonref

from .api import Function


#================ User-Defined Tools =================#

class Tools(ABC):
    '''A (fake) interface for any tool (function/ToolList/Toolkit) that can be used by the assistant.'''
    tool_enabled: bool # whether this tool is enabled
    schema: list[dict] # list of OpenAI function schemas
    lookup: dict[str, Callable] # dict of tool name to function implementation


class ToolList(Tools):
    '''A simple collection of tools/toolkits'''
    def __init__(self, *tools: Tools, tool_enabled = True):
        self.tools = list(tools)
        self.tool_enabled = tool_enabled

    @property
    def schema(self) -> list[dict]:
        '''list of OpenAI function schemas'''
        if not self.tool_enabled:
            return []
        
        return list(chain(*[t.schema for t in self.tools if t.tool_enabled]))

    @property
    def lookup(self) -> dict[str, Callable]:
        '''dict of TOOL NAME to argument-validated function'''
        if not self.tool_enabled:
            return {}
        
        lookups = [t.lookup for t in self.tools if t.tool_enabled]
        assert len(set(chain(*[lookup.keys() for lookup in lookups]))) == sum([len(lookup) for lookup in lookups]), "Duplicate tool names detected!"
        return {
            k: v
            for lookup in lookups
            for k, v in lookup.items()
        }


class Toolkit(Tools):
    '''
    A base class for a collection of tools and their shared states.
    Simply inherit this class and mark your methods as tools with the `@function_tool` decorator.
    After instantiating your toolkit, you can either:
    - [Code]: Simply use the functions as normal, e.g. `toolkit.my_tool(**args)`
    - [Model]: Use the `toolkit.lookup` dict to call the function by name, e.g. `toolkit.lookup['my_tool'](args)`
    '''
    def __init__(self):
        self.tool_enabled = True

    @property
    def schema(self) -> list[dict]:
        '''list of OpenAI function schemas'''
        return list(chain(*[tool.schema for tool in self._function_tools.values()]))
    
    @property
    def lookup(self) -> dict[str, Callable]:
        '''dict of TOOL NAME to argument-validated function'''
        return {
            tool.name: self._with_self(tool.validate_and_call)
            for tool in self._function_tools.values()
        }
    
    @property
    def _function_tools(self) -> dict[str, Callable]:
        '''dict of RAW FUNCTION NAME to function'''
        return {
            attr: getattr(self, attr)
            for attr in dir(type(self))
            if not isinstance(getattr(type(self), attr), property) # ignore properties to prevent infinite recursion
            and getattr(getattr(self, attr), 'tool_enabled', False)
        } if self.tool_enabled else {}
    
    # util to prevent late-binding of func in a dict comprehension
    def _with_self(self, func: Callable):
        '''Make a function which automatically receives self as the first argument'''
        @wraps(func)
        def wrapper(kwargs: dict[str, any]):
            return func({'self': self, **kwargs})
        return wrapper




#================ Decorators =================#

def function_tool(function=None, *, name: Optional[str] = None, require_doc: bool = True, json_schema: Optional[dict] = None, in_thread: Optional[bool] = None):
    '''
    Simple decorator that:
    - Marks a function as a tool and enables it: `func.tool_enabled = True`
    - Attaches a lookup dict for OpenAI: `func.lookup = {func.name: func}`
    - Attaches a list of OpenAI tool schema: `func.schema = [{...}]`
    - Attaches a pydantic argument validator: `func.validator`
    - Attaches a validate and call function: `func.validate_and_call(args)`
    - If a name is not specified, use the function name as the tool name
    '''
    def decorator(func):
        def validate_and_call(args: dict) -> str:
            try:
                args_without_self = {k: v for k, v in args.items() if k != 'self'}
                func.validator(**args_without_self)
            except ValidationError as e:
                return f'Invalid Argument: {e}'
            return func(**args)
        
        func.name = name or func.__name__
        func.tool_enabled = True

        if json_schema:
            # take the given json schema
            func.validator = validator_from_schema(json_schema, name=func.name)
            func.schema = [schema_to_openai_func(json_schema)]
        else:
            # parse the docstring and create a pydantic model as validator
            model, func.validator = validator_from_doc(func, name=func.name, require_doc=require_doc)
            func.schema = [schema_to_openai_func(model)]

        func.validate_and_call = validate_and_call
        func.lookup = {func.name: func.validate_and_call}
        func.validate_and_call.in_thread = not asyncio.iscoroutinefunction(func) if in_thread is None else in_thread
        return func
    
    if function: # user did `@function_tool`, i.e. we were used directly as a decorator
        return decorator(function)
    else: # user did `@function_tool()` or `@function_tool(name='foo')`
        return decorator


def fail_with_message(message, include_exception=True, logger: Callable = print):
    '''A decorator that catches exceptions from synchronous and asynchronous functions and returns the given message instead. Useful for agent tools.'''

    def log_exception(func, args, kwargs, e):
        if logger:
            logger(f"Tool call {func.__name__}({', '.join(list(map(repr, args)) + [f'{k}={repr(v)}' for k,v in kwargs.items()])}) failed: {e}")
        return message + (f': {str(e)}' if include_exception else '')

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    return log_exception(func, args, kwargs, e)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    return log_exception(func, args, kwargs, e)
            return sync_wrapper
    return decorator



#================ Model =================#

async def call_requested_function(call_request: Function, func_lookup: dict[str, Callable]):
    '''
    Call the requested function generated by the model.
    '''
    # parse function call
    func_name = call_request.name
    arguments = call_request.arguments

    if func_name not in func_lookup:
        return f"Error: Function {func_name} does not exist."
    try:
        args = json.loads(arguments)
    except Exception as e:    
        return f"Error: Failed to parse arguments, make sure your arguments is a valid JSON object: {e}"

    # call function
    try:
        f = func_lookup[func_name]
        if not getattr(f, 'in_thread', True):
            print("awaiting coroutine")
            return await f(args)
        else:
            print("awaiting thread")
            return await asyncio.to_thread(f, args)
    except Exception as e:
        return f"Error: {e}"


#================ Utils =================#

class ValidationError(BaseException):
    pass

# Function -> Pydantic model
def validator_from_doc(func: Callable, name: Optional[str] = None, require_doc = True) -> type[BaseModel]:
    '''
    Convert a function to a pydantic model for validation of function parameters.

    Args:
        func: function to convert
        name: name of the model, defaults to func.__name__

    Returns:
        pydantic model
    '''
    sig: inspect.Signature = inspect.signature(func)
    doc = parse(inspect.getdoc(func) or "")

    # get function name
    name = name or func.__name__

    # get general function description
    summary = f"{doc.short_description or ''}\n{doc.long_description or ''}".strip()

    # {param_name: param_description}
    param_descriptions = {
        doc_param.arg_name: doc_param.description
        for doc_param in doc.params
    }

    # convert function parameters to pydantic model fields
    model_fields = {}

    for param in sig.parameters.values():
        if param.name in ["self", "return", "return_type"]:
            continue

        if require_doc:
            assert param.name in param_descriptions, (f"Missing description for parameter {param.name} in {func.__name__}'s docstring!")
        
        assert param.annotation != inspect.Parameter.empty, (f"Missing type annotation for parameter {param.name} in {func.__name__}!")
            
        model_fields[param.name] = (
            param.annotation, # type
            Field(
                title=None,
                description=param_descriptions.get(param.name),
                default=
                    ...
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
            err = "\n".join([l for l in str(e).splitlines() if "further information" not in l.lower()])
            raise ValidationError(err)
        
    return model, validate_pydantic


# JSON Schema -> JSON Validator that raises ValidatonError just like Pydantic Model
def validator_from_schema(json_schema: dict, name: Optional[str] = None) -> callable:
    if name:
        json_schema['title'] = name

    # Validate json against schema
    def validate_json(**state):
        v = Draft202012Validator(json_schema)
        if errors := '\n'.join({e.message for e in v.iter_errors(state)}):
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
    remove_title(schema['properties'])

    # Construct the OpenAI function schema format
    return {
        'type': 'function',
        "function": {
            'name': schema['title'],
            **({'description': d} if (d:=schema.get('description')) else {}),
            'parameters': {
                'type': 'object',
                'properties': schema['properties'],
                'required': schema.get('required', [])
            }
        }
    }


def to_nested_schema(schema: dict, no_title=True) -> dict:
    '''nested json schema rather than refs'''
    schema = jsonref.loads(json.dumps(schema), proxies=False)
    if 'definitions' in schema:
        schema.pop('definitions')
    if '$defs' in schema:
        schema.pop('$defs')
    if no_title:
        remove_title(schema)
    return schema

def remove_title(d) -> dict | list:
    if isinstance(d, dict):
        if 'title' in d and type(d['title']) == str:
            d.pop('title')
        for v in d.values():
            remove_title(v)
    elif isinstance(d, list):
        for v in d:
            remove_title(v)
    return d
