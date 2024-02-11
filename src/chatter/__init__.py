"""Clients for chat models
"""

import typing

import openai as _openai
from openai.types.chat import ChatCompletionUserMessageParam

CompletionParametersT_contra = typing.TypeVar(
    "CompletionParametersT_contra",
    bound="CompletionParameters",
    contravariant=True,
)


class Message(str):
    """Placeholder class for messages. Should include role etc.
    """


class Completion(dict):
    """Placeholder class for completions. Should include choices etc.
    """


class CompletionChunk(dict):
    """Placeholder class for completion chunks. Should include choices etc.
    """


class CompletionParameters(typing.TypedDict, total=False):
    """Base class for model specific parameters
    """


class TokenCredential:
    """Placeholder class for token credentials. Either azure.core.credentials.TokenCredential
    azure identity's token provider callable may be viable candidates...
    """

    def __call__(self) -> str:
        return "dummytoken"  # Hackhack - should be a real token provider callable...


class BaseChatClient(typing.Generic[CompletionParametersT_contra]):
    """Chat client to talk to unknown chat models.
    """
    def __init__(
        self,
        endpoint: str,
        credentials: str | TokenCredential,
        *,
        defaults: CompletionParametersT_contra | None = None
    ):
        self._defaults = defaults
        self._endpoint = endpoint
        self._credentials = credentials

        credentials_param: typing.Any = (
            {  # Hackhack - we know that these parameters are going to be safe to use
                "api_key": credentials
            }
            if isinstance(credentials, str)
            else {"azure_ad_token_provider": credentials}
        )

        self._client = _openai.AzureOpenAI(
            base_url=endpoint,
            api_version="2023-12-01-preview",
            **credentials_param
        )

    def with_parameters(self, parameters: CompletionParametersT_contra) -> typing.Self:
        """Create a copy of the current client with modified default parameters
        """
        if self._defaults:
            keys = set(self._defaults.keys()) & set(parameters.keys())
            updated_params = typing.cast(
                CompletionParametersT_contra,
                {key: parameters.get(key, self._defaults.get(key)) for key in keys},
            )
        else:
            updated_params = parameters
        return self.__class__(
            self._endpoint, self._credentials, defaults=updated_params
        ) # Hackhack - should share underlying client/connection/connection pool etc.

    def complete(
        self,
        messages: list[Message],
        model: str,
        *,
        parameters: CompletionParametersT_contra | None = None
    ) -> Completion:
        """Complete a set of messages using the model
        """
        response = self._client.chat.completions.create(
            messages=[
                ChatCompletionUserMessageParam(role="user", content=message)
                for message in messages
            ],
            model=model,
            extra_body=parameters or {},
        )
        return Completion(response.model_dump())
