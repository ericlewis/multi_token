from multi_token.language_models.mistral import MistralLMMForCausalLM
from multi_token.language_models.qwen2 import Qwen2LMMForCausalLM

LANGUAGE_MODEL_CLASSES = [MistralLMMForCausalLM, Qwen2LMMForCausalLM]
LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}