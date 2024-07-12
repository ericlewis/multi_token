from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from multi_token.language_models.base_model import (
    LMMMetaModel,
    LMMMetaForCausalLM,
)

class Qwen2LMMConfig(PretrainedConfig):
    model_type = "qwen2-lmm"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add any Qwen2-specific configuration parameters here

class Qwen2LMMModel(LMMMetaModel, PreTrainedModel):
    config_class = Qwen2LMMConfig

    def __init__(self, config: Qwen2LMMConfig):
        super().__init__(config)
        self.qwen2_model = AutoModelForCausalLM.from_pretrained(config.name_or_path, trust_remote_code=True).model

    def forward(self, *args, **kwargs):
        return self.qwen2_model(*args, **kwargs)

class Qwen2LMMForCausalLM(PreTrainedModel, LMMMetaForCausalLM):
    config_class = Qwen2LMMConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2LMMModel(config)
        self.lm_head = self.model.qwen2_model.lm_head
        self.modalities = None

    def get_model(self) -> "Qwen2LMMForCausalLM":
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, **kwargs
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        modality_inputs=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None:
            raise ValueError("inputs_embeds not supported")

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            **(modality_inputs or {}),
        }

        return model_inputs

AutoConfig.register("qwen2-lmm", Qwen2LMMConfig)
AutoModelForCausalLM.register(Qwen2LMMConfig, Qwen2LMMForCausalLM)