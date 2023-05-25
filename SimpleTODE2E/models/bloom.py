from transformers import BloomForCausalLM


class BloomTOD(BloomForCausalLM):
    def __init__(self, hf_cfg, cfg=None):
        super(BloomTOD, self).__init__(hf_cfg)

    # Directly from GPT2 Batch Decoding
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = None # Eternally None for batch decoding

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self, input_ids, position_ids, attention_mask, labels=None,
        past_key_values=None, return_dict=None, output_attentions=None, 
        use_cache=False, output_hidden_states=None,
    ):
        transformer_outputs = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

        return transformer_outputs
