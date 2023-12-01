from typing import Type

import json
import jsonref
from pydantic import BaseModel, Field, ValidationError

# Interface for a tool, supports converting into a OpenAI Tool Scheme, validating from JSON, and actually calling from parsed args
class FunctionTool(BaseModel):
    '''
    Whatever you write here will be passed to the OpenAI model as a description of this function tool; you should provide a detailed description of the tool.
    '''
#================ MUST OVERRIDE =================#
    # Define args here, use = Field(description='field description') to provide a description
    # my_arg1: str = Field(..., description='my_arg1 description')
    # my_arg2: int = Field(..., description='my_arg2 description')

    # What actually gets called when the model calls this function
    def __call__(args, state: any = None) -> str:
        '''
        All the args you defined above are available here under args (which is just a renamed `self`), in order to ensure no naming conflicts with the state.
        if you need to access the state (e.g. given by the parent Toolkit class), you can access it via the state parameter, with the type of your choice.
        '''
        # return 'success'
        raise NotImplementedError('You must override this method')

#================ OPTIONAL OVERRIDE =================#
    @classmethod
    def openai_name(cls) -> str:
        '''Name of this function as exposed to the OpenAI Model, override as needed'''
        return cls.__name__
    
#================ FOR Toolkit CLASS =================#
    @classmethod
    def to_openai(cls) -> dict:
        schema = schema_to_openai_func(cls)
        schema['function']['name'] = cls.openai_name()
        return schema
    
    @classmethod
    def validate_and_call(cls, args: dict, state: any) -> str:
        '''Validate the args using pydantic and call the function'''
        try:
            # first create an instance, which validates and saves the args
            _self = cls(**args)
        except ValidationError as e:
            return f'Error: Invalid Arguments {e}'
        # then call the actual function, which can access the args via self and the state via state
        return _self(state)


# Toolkit is a group of tools and lets you define shared states
class Toolkit:
    '''
    A toolkit is a class you should override for a collection of tools and a lookup table for it. It also defines the shared states between the tools.
    '''
    def __init__(self, tools: list[Type[FunctionTool]]):
        self.tools = tools

        # define your states needed by the tools here
        self.state = None

    def to_openai(self) -> dict:
        '''
        Create a schema for all the tools that you can directly pass to the OpenAI API as tools
        '''
        return [tool.to_openai() for tool in self.tools]

    def to_tool_lookup(self) -> dict[str, Type[FunctionTool]]:
        '''
        Create a lookup table for tools, which you can lookup by name (str) and call with the same expected keyword arguments as the model would.
        '''
        return {
            tool.openai_name(): self._func_with_state(tool)
            for tool in self.tools
        }
    
    def _func_with_state(self, tool: Type[FunctionTool]):
        '''
        This lambda is essentially just factored out to prevent late-binding problem of using tool directly in a for loop (which would always use the last tool)
        '''
        return lambda **kwargs: tool.validate_and_call(kwargs, self.state)
    


#================ Utils =================#

# Pydantic -> OpenAI function schema
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

def to_nested_schema(model: Type[BaseModel], no_title=True) -> dict:
    '''nested json schema rather than refs'''
    schema = jsonref.loads(json.dumps(model.model_json_schema()), proxies=False)
    if 'definitions' in schema:
        schema.pop('definitions')
    if '$defs' in schema:
        schema.pop('$defs')
    if no_title:
        remove_title(schema)
    return schema

def schema_to_openai_func(schema: dict | Type[BaseModel], nested=True) -> dict:
    if not isinstance(schema, dict) and issubclass(schema, BaseModel):
        if nested:
            schema = to_nested_schema(schema, no_title=False)
        else:
            schema = schema.model_json_schema()

    # Convert properties
    remove_title(schema['properties'])

    # Construct the OpenAI function schema format
    return {
        'type': 'function',
        "function": {
            'name': schema.get('title', ''),
            'description': schema.get('description', ''),
            'parameters': {
                'type': 'object',
                'properties': schema['properties'],
                'required': schema.get('required', [])
            }
        }
    }
