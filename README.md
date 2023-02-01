# ai21-py
ai21 python client

# Supported APIs

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

In the following sections, a description of the model parameters is given for each API. 
Note that this package supports both the PEP8 `snake_case` parameter naming convention 
and the `camelCase` convention that the AI21 APIs use. When passing in your model parameters, you can use
either version but to adhere to PEP8 standard in the functions defined herein, we use the `snake_case`
standard.

# api = 'complete'

`prompt`: the lead-in to the completion. This is where few-shot learning can be done to provide a few examples of 
type oc completion you are seeking.

`num_results|numResults`: the number of completions to return

`max_tokens|maxTokens`: The maximum number of tokens to generate. J1 models have a capacity of 2048 tokens in total,
including both the prompt and the generated completion. This corresponds to 2300-2500 English words on average.

`temperature`: Controls sampling randomness. Increasing the temperature tends to result in more varied and
creative completions, whereas decreasing it results in more stable and repetitive completions. A temperature of
zero effectively disables random sampling and makes the completions deterministic. Setting temperature to 1.0
samples directly from the model distribution. Lower (higher) values increase the chance of sampling higher (lower)
probability tokens. A value of 0 essentially disables sampling and results in greedy decoding, where the most
likely token is chosen at every step. Must be on interval [0, 5].

`top_returns|topKReturn`: Return the top-K alternative tokens. When using a non-zero value, the
response includes the string representations and logprobs for each of the top-K alternatives at each position,
in the prompt and in the completions. Must be on the interval [0, 64].

`top_percentile|topP`: The percentile of probability from which tokens are sampled. A value lower than 1.0 means
that only the corresponding top percentile of options is considered, and that less likely options will not be
generated, resulting in more stable and repetitive completions. For example, a value of 0.9 will only consider
tokens comprising the top 90% probability mass. Must be on the interval [0, 1].

`count_penalty|countPenalty`: Applies a bias against generating tokens that appear in the prompt or in the completion,
proportional to the number of respective appearances.

`frequency_penalty|frequencyPenalty`: Applies a bias against generating tokens that appeared in the prompt or in the
completion, proportional to the frequency of respective appearances in the text.

`presence_penalty|presencePenalty`: Applies a fixed bias against generating tokens that appeared at least once in the prompt
or in the completion.

`stop_sequences|stopSequences`: Stops decoding if any of the strings is generated. For example, to stop at a comma or a
new line use [".", "\n"]. The decoded result text will not include the stop sequence string, but it will be
included in the raw token data, which can also continue beyond the stop sequence if the sequence ended in the
middle of a token.

`model`: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.


# api = 'summarize'

`text`: text to summarize

# api = 'rewrite'

`text`: text to rewrite

`intent`: style of rewrite. Supported types are 'general', 'long', 'short', 'formal', 'casual'

