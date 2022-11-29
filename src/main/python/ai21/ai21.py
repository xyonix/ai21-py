"""Utilities to interact with AI21 Large Language Model APIs"""

import requests
from ai21 import llm_prompts
AI21_API_KEY_PATH='~/creds/.ai21'
from typing import List, Dict


AI21_PARAM_LIMITS = {
    'MAX_TOKENS': (1, 2048),
    'TEMPERATURE': (0, 5),
    'TOP_RETURNS': (0, 64),
    'TOP_PERCENTILE': (0, 1),
}

SUPPORTED_MODELS = ['j1-large', 'j1-grande', 'j1-jumbo']
SUPPORTED_MODEL_TYPES = ['summarization', 'completion']


def check_api_params(max_tokens: int, temperature: float, top_returns: int, top_percentile: float,
                     model: [str, None], model_type: [str, None]):
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
    @param model_type: type of model. Choices are 'summarization', 'completion'.
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
    if model_type is not None and model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError('model_type "%s" be one of %s' % (model_type, str(SUPPORTED_MODEL_TYPES)))


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


def get_authorization_string(api_key_path) -> str:
    """
    Obtain the AI21 authorization string.
    @param api_key_path: The path to a local file containing a AI21_API_KEY environment variable. Your API key can be
    generated by visiting https://studio.ai21.com/account/account.
    @return: AI21 API authorization string.
    """
    import os
    import dotenv
    dotenv.load_dotenv(os.path.expanduser(api_key_path), override=True)
    return f"Bearer {os.environ['AI21_API_KEY']}"


def get_api_url(model_type: str = 'complete', model: str = 'j1-large') -> str:
    """
    Obtain the AI21 API URL.
    @param model_type: type of model. Choices are 'summarization', 'completion'.
    @param model: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.
    @return: AI21 API URL
    """
    model_type = model_type.lower()
    model = model.lower()

    if model not in SUPPORTED_MODELS:
        raise ValueError('Unsupported large language model. Must be one of %s' % SUPPORTED_MODELS)

    if model_type == 'summarization':
        url = 'https://api.ai21.com/studio/v1/experimental/summarize'
    elif model_type == 'completion':
        url = f"https://api.ai21.com/studio/v1/{model.lower()}/complete"
    else:
        raise ValueError('Unsupported model type. Must be one of %s' % SUPPORTED_MODEL_TYPES)

    return url


def api_request(prompt: str, num_results: int = 1, max_tokens: int = 16, temperature: float = 1.0,
                top_returns: int = 0, top_percentile: float = 1.0,
                count_penalty: Dict = None, frequency_penalty: Dict = None, presence_penalty: Dict = None,
                stop_sequences: List[str] = ("##",), model_type: str = 'completion', model: str = 'j1-large',
                api_key_path: str = AI21_API_KEY_PATH, config=None) -> requests.models.Response:
    """
    Post JSON to a REST API endpoint for AI21's Jurassic-1 (J1) NLP large language models.
    @param prompt: Prompt
    @param num_results:
    @param max_tokens: The maximum number of tokens to generate. J1 models have a capacity of 2048 tokens in total,
    including both the prompt and the generated completion. This corresponds to 2300-2500 English words on average.
    @param temperature: Controls sampling randomness. Increasing the temperature tends to result in more varied and
    creative completions, whereas decreasing it results in more stable and repetitive completions. A temperature of
    zero effectively disables random sampling and makes the completions deterministic. Setting temperature to 1.0
    samples directly from the model distribution. Lower (higher) values increase the chance of sampling higher (lower)
    probability tokens. A value of 0 essentially disables sampling and results in greedy decoding, where the most
    likely token is chosen at every step. Must be on interval [0, 5].
    @param top_returns: Return the top-K alternative tokens. When using a non-zero value, the
    response includes the string representations and logprobs for each of the top-K alternatives at each position,
    in the prompt and in the completions. Must be on the interval [0, 64].
    @param top_percentile: The percentile of probability from which tokens are sampled. A value lower than 1.0 means
    that only the corresponding top percentile of options are considered, and that less likely options will not be
    generated, resulting in more stable and repetitive completions. For example, a value of 0.9 will only consider
    tokens comprising the top 90% probability mass. Must be on the interval [0, 1].
    @param count_penalty: Applies a bias against generating tokens that appear in the prompt or in the completion,
    proportional to the number of respective appearances.
    @param frequency_penalty: Applies a bias against generating tokens that appeared in the prompt or in the
    completion, proportional to the frequency of respective appearances in the text.
    @param presence_penalty: Applies a fixed bias against generating tokens that appeared at least once in the prompt
    or in the completion.
    @param stop_sequences: Stops decoding if any of the strings is generated. For example, to stop at a comma or a
    new line use [".", "\n"]. The decoded result text will not include the stop sequence string, but it will be
    included in the raw token data, which can also continue beyond the stop sequence if the sequence ended in the
    middle of a token.
    @param model_type: type of model. Choices are 'summarization', 'completion'.
    @param model: AI21 large language model to use. Choices are 'j1-large', 'j1-grande', 'j1-jumbo'.
    @param api_key_path: Path to a local file containing a AI21_API_KEY environment variable. Your API key can be
    generated by visiting https://studio.ai21.com/account/account.
    @param config: Alternative to specifying input parameters individually, the collection of parameters can be
    defined in a config.py Python file (for example) and sourced via an 'import config as cfg' call. The cfg object
    can then be passed to this function. By default this parameter is None, meaning that you will need to specify all
    of the other parameters explicitly. If not None, various parameters are expected to be defined in the config
    file and used to make the REST call: MAX_TOKENS, TEMPERATURE, TOP_RETURN, TOP_PERCENTILE, COUNT_PENALTY,
    FREQUENCY_PENALTY, PRESENCE_PENALTY, STOP_SEQUENCES.
    @return: response for the AI21 post request.
    """
    if config is None:

        if count_penalty is None:
            count_penalty = DEFAULT_PENALTY
        if frequency_penalty is None:
            frequency_penalty = DEFAULT_PENALTY
        if presence_penalty is None:
            presence_penalty = DEFAULT_PENALTY

        check_api_params(max_tokens, temperature, top_returns, top_percentile, model, model_type)

        response = requests.post(get_api_url(model_type=model_type, model=model),
                                 headers={"Authorization": get_authorization_string(api_key_path)},
                                 json={
                                     "prompt": prompt,
                                     "numResults": num_results,
                                     "maxTokens": max_tokens,
                                     "temperature": temperature,
                                     "topKReturn": top_returns,
                                     "topP": top_percentile,
                                     "countPenalty": count_penalty,
                                     "frequencyPenalty": frequency_penalty,
                                     "presencePenalty": presence_penalty,
                                     "stopSequences": stop_sequences
                                 }
                                 )
    else:

        check_api_params(config.MAX_TOKENS, config.TEMPERATURE, config.TOP_RETURNS, config.TOP_PERCENTILE,
                         None, None)

        response = requests.post(config.URL,
                                 headers={"Authorization": config.AUTH},
                                 json={
                                     "prompt": prompt,
                                     "numResults": num_results,
                                     "maxTokens": config.MAX_TOKENS,
                                     "temperature": config.TEMPERATURE,
                                     "topKReturn": config.TOP_RETURNS,
                                     "topP": config.TOP_PERCENTILE,
                                     "countPenalty": config.COUNT_PENALTY,
                                     "frequencyPenalty": config.FREQUENCY_PENALTY,
                                     "presencePenalty": config.PRESENCE_PENALTY,
                                     "stopSequences": config.STOP_SEQUENCES
                                 })

    return response


def get_response_completion(response: requests.models.Response, completion_index: int = 0) -> str:
    """
    Extract the completion from the response object.
    @param response: Response from api_request() call.
    @param completion_index: The index of te completion to extract. Typically, there will be only one completion
    returned in the response and so the default index is 0.
    @return: The completion of returned by the J1 large language model.
    """
    data = response.json()
    completion = data['completions'][completion_index]['data']['text'].strip()
    return completion


def sentence_sentiment_labeler(sentence, max_tokens=114, temperature=0.33, model='j1-grande'):
    prompt_preface = llm_prompts.SENTIMENT_SENTENCE_LABELING['preface'].strip('\n')
    prompt_end = llm_prompts.SENTIMENT_SENTENCE_LABELING['end']
    sentence = sentence.strip().strip('\n')
    prompt = f"{prompt_preface}{sentence}{prompt_end}"
    resp = api_request(prompt, num_results=1, max_tokens=max_tokens, temperature=temperature,
                       top_returns=0, top_percentile=1.0,
                       # presence_penalty=get_penalty(scale=5),
                       # frequency_penalty=get_penalty(scale=5),
                       stop_sequences=["##"], model_type='completion', model=model)
    completion = get_response_completion(resp)
    delimeter = ', '
    return delimeter.join(list(set(completion.split(delimeter))))
