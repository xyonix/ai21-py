from config import AI21_API_KEY_PATH
from ai21 import get_authorization_string, get_penalty, get_api_url

MODEL = 'j1-jumbo'
TEMPERATURE = 0.3
MAX_TOKENS = 45
TOP_RETURNS = 0
TOP_PERCENTILE = 0.98
STOP_SEQUENCES = ["##"]
AUTH = get_authorization_string(api_key_path=AI21_API_KEY_PATH)
URL = get_api_url(model_type='completion', model=MODEL)

COUNT_PENALTY = get_penalty()
FREQUENCY_PENALTY = get_penalty()
PRESENCE_PENALTY = get_penalty()
