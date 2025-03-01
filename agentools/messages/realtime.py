# from typing import override
# from .core import MessageHistory, ChatCompletionMessage


# class RealtimeHistory(MessageHistory):
#     @property
#     def history(self):
#         return self._history

#     @override
#     async def append(self, message: dict | ChatCompletionMessage):
#         """Add a message to the history"""
#         message = self.ensure_dict(message)
#         self.history.append(message)

#     @override
#     async def reset(self):
#         """Reset the history"""
