# Experiments with chat and tracing

Try out the experience with various API patterns for clients.

Example usage(s):

```python
import chatter
import chatter.llama
import chatter.openai

client = chatter.openai.ParametersOpenAIClient(endpoint='zzz://openai.com',
                                               credentials="dummy",
                                               defaults={"temperature": 0.5})

completion = client.complete(
    messages = [chatter.Message("hello")],
    model="gpt-4",
    parameters = chatter.openai.OpenAIChatParameters(logprobs=True)
)

llamaclient = chatter.llama.ParametersLlamaClient(endpoint='zzz://llama.com', credentials="dummytoken")
llamaclient.complete([], "", parameters = { "seed": "123" })
```

## Setup

pip install git+https://git@github.com/johanste/chatter#egg=chatter
