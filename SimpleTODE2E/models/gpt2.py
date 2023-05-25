from transformers import GPT2LMHeadModel


class GPT2TOD(GPT2LMHeadModel):
    def __init__(self, hf_cfg, cfg=None):
        super(GPT2TOD, self).__init__(hf_cfg)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        del kwargs['position_ids']
        kwargs['position_ids'] = None  # To be figured out using attention mask
        gpt_ret = super().prepare_inputs_for_generation(input_ids, **kwargs)

        ret = dict()
        ret["input_ids"] = gpt_ret['input_ids']
        ret['position_ids'] = gpt_ret['position_ids']
        ret['attention_mask'] = gpt_ret['attention_mask']
        ret['use_cache'] = gpt_ret['use_cache']
        ret['past_key_values'] = gpt_ret['past_key_values']

        return ret

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
