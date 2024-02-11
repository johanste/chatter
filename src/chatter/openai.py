"""OpenAI bindings for chat models
"""

from typing import Dict, List, Optional, Union, Unpack

import chatter


class OpenAIChatParameters(chatter.CompletionParameters, total=False):
    """Model specific parameters for the OpenAI completion API."""

    frequency_penalty: Optional[float]
    """Frequency penalty to reduce the likelihood of repeating the same response.
    """

    function_call: Dict[str, object]
    """Function call to be made to the model."""
    
    functions: List[object]
    logit_bias: Optional[Dict[str, int]]
    logprobs: Optional[bool]
    max_tokens: Optional[int]
    n: Optional[int]
    presence_penalty: Optional[float]
    response_format: object
    seed: Optional[int]
    stop: Union[Optional[str], List[str]]
    stream: Optional[bool]
    temperature: Optional[float]
    tool_choice: Dict[str, object]
    tools: List[Dict[str, object]]
    top_logprobs: Optional[int]
    top_p: Optional[float]
    timeout: float | None


class ParametersOpenAIClient(chatter.BaseChatClient[OpenAIChatParameters]):
    """Model specific parameters for the OpenAI chat completion models


    This client is used to demonstrate that the parameters can be passed as
    a dictionary.
    """


class KeywordsOpenAIClient(chatter.BaseChatClient[OpenAIChatParameters]):
    """Model specific parameters for the OpenAI chat completion models

    This client is used to demonstrate that the parameters can be passed as
    keyword arguments.
    """

    def complete(
        self,
        messages: List[chatter.Message],
        model: str,
        **kwargs: Unpack[OpenAIChatParameters]
    ) -> chatter.Completion:
        return super().complete(messages, model, parameters=kwargs)
