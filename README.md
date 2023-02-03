# ai21-py
ai21 python client

# requirements

This package requires Python 3.9+ and that have obtained a working API key from AI21.

# supported APIs

There are multiple AI21 APIs that are currently supported. You will generally use the `api_request` 
function to hit the AI21 API of choice, then you can obtain the results with the convenience function 
`get_response_text`. Here are more details on the _**kwargs_ parameter of the `api_request` function. 
The default parameter values for each of these APIs can be found by calling the `get_api_params` function 
with the API you wish to hit, e.g., 

```python
get_api_params(api='complete')
```

Then, when you call `api_request`, you only need to specify the parameters that are different from
the defaults, since the user-specified values via the _**kwargs_ parameter overwrites the corresponding 
default values.

## api = 'complete'

`prompt`: the lead-in to the completion. This is where few-shot learning can be done to provide a few examples of 
type oc completion you are seeking.

`num_results`: the number of completions to return

`max_tokens`: The maximum number of tokens to generate. J1 models have a capacity of 2048 tokens in total,
including both the prompt and the generated completion. This corresponds to 2300-2500 English words on average.

`temperature`: Controls sampling randomness. Increasing the temperature tends to result in more varied and
creative completions, whereas decreasing it results in more stable and repetitive completions. A temperature of
zero effectively disables random sampling and makes the completions deterministic. Setting temperature to 1.0
samples directly from the model distribution. Lower (higher) values increase the chance of sampling higher (lower)
probability tokens. A value of 0 essentially disables sampling and results in greedy decoding, where the most
likely token is chosen at every step. Must be on interval [0, 5].

`top_returns`: Return the top-K alternative tokens. When using a non-zero value, the
response includes the string representations and logprobs for each of the top-K alternatives at each position,
in the prompt and in the completions. Must be on the interval [0, 64].

`top_percentile`: The percentile of probability from which tokens are sampled. A value lower than 1.0 means
that only the corresponding top percentile of options is considered, and that less likely options will not be
generated, resulting in more stable and repetitive completions. For example, a value of 0.9 will only consider
tokens comprising the top 90% probability mass. Must be on the interval [0, 1].

`count_penalty`: Applies a bias against generating tokens that appear in the prompt or in the completion,
proportional to the number of respective appearances.

`frequency_penalty`: Applies a bias against generating tokens that appeared in the prompt or in the
completion, proportional to the frequency of respective appearances in the text.

`presence_penalty`: Applies a fixed bias against generating tokens that appeared at least once in the prompt
or in the completion.

`stop_sequences`: Stops decoding if any of the strings is generated. For example, to stop at a comma or a
new line use [".", "\n"]. The decoded result text will not include the stop sequence string, but it will be
included in the raw token data, which can also continue beyond the stop sequence if the sequence ended in the
middle of a token.

`model`: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.


## api = 'summarize'

`text`: text to summarize

## api = 'rewrite'

`text`: text to rewrite

`intent`: style of rewrite. Supported types are 'general', 'long', 'short', 'formal', 'casual'

# sample usage

Start by importing relevant functions into your environment:
```python
from ai21.api import api_request, get_response_text
```

Then you can use those functions to obtain response and content form various AI21 APIs;

## api='complete'
```python
response, params = api_request(api='complete', model='j1-large', prompt="What is larger: a pizza or the moon?",
                               max_tokens=100, temperature=0.9, top_percentile=0.5)
completion = get_response_text(response)
print(f'response.ok: {response.ok}\n\nparams: {params}\n\ncompletion: {completion}')
```

## api='summarize'
```python
text = """
Elizabeth II (Elizabeth Alexandra Mary; 21 April 1926 – 8 September 2022) was Queen of the United Kingdom 
and other Commonwealth realms from 6 February 1952 until her death in 2022. She was queen regnant of 32 
sovereign states during her lifetime and 15 at the time of her death.[a] Her reign of 70 years and 214 
days is the longest of any British monarch and the longest recorded of any female head of state in history.
Elizabeth was born in Mayfair, London, as the first child of the Duke and Duchess of York (later King George VI 
and Queen Elizabeth). Her father acceded to the throne in 1936 upon the abdication of his brother, 
King Edward VIII, making Elizabeth the heir presumptive. She was educated privately at home and began to 
undertake public duties during the Second World War, serving in the Auxiliary Territorial Service. 
In November 1947, she married Philip Mountbatten, a former prince of Greece and Denmark, and their 
marriage lasted 73 years until his death in April 2021. They had four children: Charles, Anne, Andrew, and Edward.
"""

response, params = api_request(api='summarize', text=text.replace('\n', ' ').strip())
completion = get_response_text(response)
print(f'response.ok: {response.ok}\n\nparams: {params}\n\ncompletion: {completion}')
```

## api='rewrite'
```python
text = """When two particles, such as a pair of photons or electrons, become entangled, they remain connected 
even when separated by vast distances."""

response, params = api_request(api='rewrite', text=text.replace('\n', ' ').strip(), intent='casual')
completion = get_response_text(response)
print(f'response.ok: {response.ok}\n\nparams: {params}\n\ncompletion: {completion}')
```