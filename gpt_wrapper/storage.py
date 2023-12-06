'''
TODO: should this even be the scope of this project?
For object storage, we offer a mixin class that can be used to mark a class as
being able to be stored in an object store.
For the backend, we use MongoDB with the async motor driver.
'''

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class JSONState(ABC):
    '''
    Mixin class for storing objects in JSON format.
    The get_state method should return a dictionary that can be serialized to JSON.
    The from_state method should take a dictionary and completely recreate the object.

    For indexing, you should define a unique id field that is used to identify the object.
    Additionally, we expose the raw collection object for more efficient updates.
    '''
    @abstractmethod
    def get_state(self) -> dict:
        '''
        Get the state of the object.
        '''
        pass

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict):
        '''
        Create an object from a state.
        '''
        pass