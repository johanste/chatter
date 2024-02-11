"""Hypothetical LLama bindings for chat models
"""

import typing

import chatter


class LlamaParameters(chatter.CompletionParameters, total=False):
    """Model specific parameters for the OpenAI completion API."""

    seed: str
    temperature: typing.Optional[float]
    logit: bool


class ParametersLlamaClient(chatter.BaseChatClient[LlamaParameters]):
    """Model specific parameters for the llama chat completion models


    This client is used to demonstrate that the parameters can be passed as
    a dictionary.
    """


class KeywordsLlamaClient(chatter.BaseChatClient[LlamaParameters]):
    """Model specific parameters for the Lllama chat completion models

    This client is used to demonstrate that the parameters can be passed as
    keyword arguments.
    """

    def complete(
        self,
        messages: typing.List[chatter.Message],
        model: str,
        **kwargs: typing.Unpack[LlamaParameters]
    ) -> chatter.Completion:
        return super().complete(messages, model, parameters=kwargs)
