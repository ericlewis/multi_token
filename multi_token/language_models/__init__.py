from multi_token.language_models.mistral import MistralLMMForCausalLM
from multi_token.language_models.qwen2 import Qwen2LMMForCausalLM
from multi_token.language_models.phi3 import Phi3LMMForCausalLM

LANGUAGE_MODEL_CLASSES = [MistralLMMForCausalLM, Qwen2LMMForCausalLM, Phi3LMMForCausalLM]
LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}