"""Utilities to interact with AI21 Large Language Model APIs"""

import requests
from ai21 import llm_prompts
from typing import Dict, Union, Tuple, List

AI21_API_KEY_PATH = '~/creds/.ai21'

AI21_PARAM_LIMITS = {
    'MAX_TOKENS': (1, 2048),
    'TEMPERATURE': (0, 5),
    'TOP_RETURNS': (0, 64),
    'TOP_PERCENTILE': (0, 1),
}

SUPPORTED_MODELS = ['j1-large', 'j1-grande', 'j1-jumbo']
SUPPORTED_API = ['summarize', 'complete', 'rewrite']
SUPPORTED_REWRITE_INTENT = ['general', 'long', 'short', 'formal', 'casual']

API_COST = {'j1-large': 0.0003, 'j1-grande': 0.0008, 'j1-jumbo': 0.0128, 'summarize': 0.005}

API_RATE_PER_1K_GEN_TOKENS = {'j1-large': 0.03, 'j1-grande': 0.08, 'j1-jumbo': 0.25}

API_SUMMARY_COST_PER_CHARACTER = 0.00002

PARAMETER_MAP = {
    'num_results': 'numResults',
    'max_tokens': 'maxTokens',
    'top_returns': 'topKReturn',
    'top_percentile': 'topP',
    'count_penalty': 'countPenalty',
    'frequency_penalty': 'frequencyPenalty',
    'presence_penalty': 'presencePenalty',
    'stop_sequences': 'stopSequences',
}


def check_api_params(max_tokens: int, temperature: float, top_returns: int, top_percentile: float,
                     model: [str, None], api: [str, None]):
    """
    Check the range of certain parameters in the AI21 API call.
    @param max_tokens: The maximum number of tokens to generate. J1 models have a capacity of 2048 tokens in total,
    including both the prompt and the generated completion. This corresponds to 2300-2500 English words on average.
    @param temperature: Controls sampling randomness. Increasing the temperature tends to result in more varied and
    creative completions, whereas decreasing it results in more stable and repetitive completions. A temperature of
    zero effectively disables random sampling and makes the completions deterministic. Setting temperature to 1.0
    samples directly from the model distribution. Lower (higher) values increase the chance of sampling higher (lower)
    probability tokens. A value of 0 essentially disables sampling and results in greedy decoding, where the most
    likely token is chosen at every step.
    @param top_returns: Return the top-K alternative tokens. When using a non-zero value, the
    response includes the string representations and logprobs for each of the top-K alternatives at each position,
    in the prompt and in the completions. Must be on the interval [0, 64].
    @param top_percentile: The percentile of probability from which tokens are sampled. A value lower than 1.0 means
    that only the corresponding top percentile of options are considered, and that less likely options will not be
    generated, resulting in more stable and repetitive completions. For example, a value of 0.9 will only consider
    tokens comprising the top 90% probability mass. Must be on the interval [0, 1].
    @param model: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.
    @param api: type of model. Choices are 'summarize', 'complete'.
    """
    if not AI21_PARAM_LIMITS['MAX_TOKENS'][0] <= max_tokens <= AI21_PARAM_LIMITS['MAX_TOKENS'][1]:
        raise ValueError('max_tokens must be on the range: %s' % str(AI21_PARAM_LIMITS['MAX_TOKENS']))
    if not AI21_PARAM_LIMITS['TEMPERATURE'][0] <= temperature <= AI21_PARAM_LIMITS['TEMPERATURE'][1]:
        raise ValueError('max_tokens must be on the range: %s' % str(AI21_PARAM_LIMITS['TEMPERATURE']))
    if not AI21_PARAM_LIMITS['TOP_RETURNS'][0] <= top_returns <= AI21_PARAM_LIMITS['TOP_RETURNS'][1]:
        raise ValueError('max_tokens must be on the range: %s' % str(AI21_PARAM_LIMITS['TOP_RETURNS']))
    if not AI21_PARAM_LIMITS['TOP_PERCENTILE'][0] <= top_percentile <= AI21_PARAM_LIMITS['TOP_PERCENTILE'][1]:
        raise ValueError('max_tokens must be on the range: %s' % str(AI21_PARAM_LIMITS['TOP_PERCENTILE']))
    if model is not None and model not in SUPPORTED_MODELS:
        raise ValueError('model "%s" be one of %s' % (model, str(SUPPORTED_MODELS)))
    if api is not None and api not in SUPPORTED_API:
        raise ValueError('api "%s" be one of %s' % (api, str(SUPPORTED_API)))


def get_penalty(scale: float = 0.0, numbers: bool = None, punctuation: bool = None, stopwords: bool = None,
                whitespace: bool = None, emojis: bool = None) -> dict:
    """
    Get penalty dictionary for AI21 API call. If the scale=0.0, the DEFAULT application (numbers, whitespace, etc.)
    values are set to False. If the scale=0.0, the DEFAULT application values are set to True. Individually,
    these application values can be set explicitly in the call, which override the DEFAULT. If no parameters are passed
    to this function, the default behavior is that no penalties are applied.
    @param scale: Controls the magnitude of the penalty. A positive penalty value implies reducing the probability of
    repetition. Larger values correspond to a stronger bias against repetition.
    @param numbers: Apply the penalty to numbers. Determines whether the penalty is applied to purely-numeric tokens,
    such as 2022 or 123. Tokens that contain numbers and letters, such as 20th, are not affected by this parameter.
    @param punctuation: Apply the penalty to punctuations. Determines whether the penalty is applied to tokens
    containing punctuation characters and whitespaces, such as ; , !!! or ▁\\[[@.
    @param stopwords: Apply the penalty to stop words. Determines whether the penalty is applied to tokens that are
    NLTK English stopwords or multi-word combinations of these words, such as are , nor and ▁We▁have.
    @param whitespace: Apply the penalty whitespaces and newlines. Determines whether the penalty is applied to the
    following tokens: '▁', '▁▁', '▁▁▁▁', '<|newline|>'
    @param emojis: Exclude emojis from the penalty. Determines whether the penalty is applied to any of approximately
    650 common emojis in the Jurassic-1 vocabulary.
    @return: dictionary of penalty terms to be used in AI21 API JSON call.
    """

    # set defaults for applyTos: if the scale=0.0, we set the default values of these False. Otherwise,
    # id the scale > 0, we set the default values to True.
    default_app = scale > 0
    if numbers is None:
        numbers = default_app
    if punctuation is None:
        punctuation = default_app
    if stopwords is None:
        stopwords = default_app
    if whitespace is None:
        whitespace = default_app
    if emojis is None:
        emojis = default_app

    return {"scale": scale,
            "applyToNumbers": numbers,
            "applyToPunctuations": punctuation,
            "applyToStopwords": stopwords,
            "applyToWhitespaces": whitespace,
            "applyToEmojis": emojis}


DEFAULT_PENALTY = get_penalty()


def get_authorization(api_key_path: str = AI21_API_KEY_PATH) -> Dict:
    """
    Obtain the AI21 authorization string.
    @param api_key_path: The path to a local file containing a AI21_API_KEY environment variable. Your API key can be
    generated by visiting https://studio.ai21.com/account/account.
    @return: AI21 API authorization.
    """
    import os
    import dotenv
    dotenv.load_dotenv(os.path.expanduser(api_key_path), override=True)
    auth_string = f"Bearer {os.environ['AI21_API_KEY']}"
    return {"Authorization": auth_string}


def get_api_url(api: str = 'complete', model: str = 'j1-large', custom_model: Union[None, str] = None,
                version: str = 'v1') -> str:
    """
    Obtain the AI21 API URL.
    @param version: AI21 standard model release version.
    @param custom_model: custom model name.
    @param api: type of model. Choices are 'summarize', 'complete'.
    @param model: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.
    @return: AI21 API URL
    """
    api = api.lower()
    model = model.lower()
    if model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported large language model "%s". Must be one of %s' % (model, SUPPORTED_MODELS))
    version = version.lower()

    api_base_url = f'https://api.ai21.com/studio/{version}'

    if api == 'summarize':
        url = f'{api_base_url}/experimental/summarize'
    elif api == 'complete':
        if custom_model is not None:
            url = f'{api_base_url}/{model}/{custom_model}/complete'
        else:
            url = f'{api_base_url}/{model}/complete'
    elif api == 'rewrite':
        url = f'{api_base_url}/experimental/rewrite'
    else:
        raise ValueError('Unsupported model type. Must be one of %s' % SUPPORTED_API)

    return url


def camel_case(params: Dict) -> Dict:
    """
    Map parameter keys in PEP8 standard (like num_results) to the camelCase names that AI21 uses in its API call
    (like numResults).
    @param params: dictionary of model parameters with keys possibly in snake_case format.
    @return: dictionary of model parameters with camelCase converted parameter keys.
    """
    new_params = params.copy()
    for k, v in params.items():
        if k in PARAMETER_MAP:
            new_key = PARAMETER_MAP[k]
            new_params[new_key] = new_params.pop(k)
    return new_params


def api_request(api: str = 'complete', model: str = 'j1-large', custom_model: Union[None, str] = None,
                api_key_path: str = AI21_API_KEY_PATH, version='v1', **kwargs) -> Tuple[requests.models.Response, Dict]:
    """
    Post JSON to a REST API endpoint for AI21's Jurassic-1 (J1) NLP large language models. Note the the kwargs keys
    passed into this function are converted to camelCase style as needed to properly interact with the AI21 API.
    @param version: AI21 Jurassic model version.
    @param api: AI21 API. Choices are 'complete', 'summary', 'rewrite'.
    @param model: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.
    @param custom_model: name of the custom model to use.
    @param api_key_path: Path to a local file containing a AI21_API_KEY environment variable. Your API key can be
    generated by visiting https://studio.ai21.com/account/account.
    @param kwargs Additional parameters used in the API call, which overwrite the defaults.
    """
    params = camel_case(get_api_params(api, **kwargs))

    try:
        response = requests.post(get_api_url(api=api, model=model, custom_model=custom_model, version=version),
                                 headers=get_authorization(api_key_path),
                                 json=params)
        response.raise_for_status()
    except requests.exceptions.HTTPError as error:
        raise ValueError(error)
    return response, params


def get_api_params(api: str, **kwargs) -> Dict:
    """
    Forms the parameter set for various supported APIs. Note that all of the parameter names specified here are in
    PEP8 snake_case standard.
    @param api: API to use, supported values are 'complete','rewrite','summary'.
    @param kwargs: parameters specified in key=value format in the call, which overwrite the default values. Note that
    the keys you specify must be supported by the API. Otherwise, an exception is thrown.
    @return: dictionary of parameters.
    """
    api = api.lower()

    if api == 'complete':

        # default geared toward single sentence summary
        params = dict(
            prompt='',
            num_results=1,
            max_tokens=20,
            temperature=0.3,
            top_returns=0,
            top_percentile=1.0,
            count_penalty=DEFAULT_PENALTY,
            frequency_penalty=get_penalty(scale=96),
            presence_penalty=get_penalty(scale=2),
            stop_sequences=(".",),
        )

    elif api == 'summarize':
        params = dict(
            text=''
        )
    elif api == 'rewrite':
        params = dict(
            text='',
            intent='general'
        )
    else:
        raise Exception('unsupported API: %s', api)

    # trim kwargs to only those seen in default keys
    trimmed_kwargs = {k: v for k, v in kwargs.items() if k in params}

    if len(trimmed_kwargs) < len(kwargs):
        user_params = set(list(kwargs.keys()))
        api_params = set(list(params.keys()))
        deleted_params = user_params.difference(api_params)
        raise ValueError(
            "User supplied parameters are not supported for '%s' API: %s. Supported parameters are: %s" % (
                api, deleted_params, api_params))

    # combine user specified with default params
    return params | trimmed_kwargs


def get_response_api(response: requests.models.Response) -> str:
    """
    Infer the API that was hit given the response object.
    @param response: Response from api_request()
    @return: API string
    """
    return [s for s in SUPPORTED_API if s in response.url][0]


def get_response_text(response: requests.models.Response, simplify: bool = True) -> Union[List[str], str]:
    """
    Extract the completion from the response object.
    @param simplify: simplify the result?
    @param response: Response from api_request() call.
    @return: The completion of returned by the J1 large language model.
    """
    data = response.json()
    api = get_response_api(response)
    if api == 'complete':
        result = [s['data']['text'].strip() for s in data['completions']]
    elif api == 'summarize':
        result = [s['text'].strip() for s in data['summaries']]
    elif api == 'rewrite':
        result = [s['text'].strip() for s in data['suggestions']]
    else:
        result = None

    if result is not None and simplify and len(result) == 1:
        result = result[0]

    return result


def sentence_sentiment_labeler(sentence, max_tokens=114, temperature=0.33, model='j1-grande'):
    prompt_preface = llm_prompts.SENTIMENT_SENTENCE_LABELING['preface'].strip('\n')
    prompt_end = llm_prompts.SENTIMENT_SENTENCE_LABELING['end']
    sentence = sentence.strip().strip('\n')
    prompt = f"{prompt_preface}{sentence}{prompt_end}"
    resp, _ = api_request(api='complete', model=model, prompt=prompt, num_results=1, max_tokens=max_tokens,
                          temperature=temperature, top_returns=0, top_percentile=1.0, stop_sequences=["##"])
    completion = get_response_text(resp)
    delimeter = ', '
    return delimeter.join(list(set(completion.split(delimeter))))


def get_response_cost(response: requests.models.Response) -> float:
    """
    Get the cost of the response returned by hitting the AI21 API
    @param response: Response from api_request() call.
    @return: fractional cost in cents
    """
    response_json = response.json()
    if 'completions' in response_json:
        num_tokens_generated = len(response_json['completions'][0]['data']['tokens'])
        model = [s for s in response.request.path_url.split('/') if s.startswith('j1') or s.startswith('summarize')][0]

        api_cost = API_COST[model]
        api_token_rate = API_RATE_PER_1K_GEN_TOKENS[model]

        # cost is two parts:
        #    api_cost: the cost just to hit the API service
        #    api_token_rate: the cost per 1000 tokens generated
        cost = round(api_cost + api_token_rate * num_tokens_generated / 1000, 4)

    elif 'summaries' in response_json:
        summary_length = len(response_json['summaries'][0]['text'])

        cost = API_COST['summarize'] + summary_length * API_SUMMARY_COST_PER_CHARACTER
    else:
        raise ValueError('response cost cannot be calculated as it is neither a completion or summarization')

    return cost
