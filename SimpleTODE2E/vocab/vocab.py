import os
import json
import logging

import numpy as np
import torch
from transformers import (
    GPT2TokenizerFast,
    BloomTokenizerFast
)


logger = logging.getLogger()

GPT2_MAXLEN = 1024
BLOOM_MAXLEN = 1526


def read_entities(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)

    process_entities = lambda x: '_'.join(x.lower().split())

    ent2type = dict()
    for etype in obj['all_entities']:
        for ent in obj['all_entities'][etype]:
            ent2type[process_entities(ent)] = etype

    return ent2type


class CausalVocab(object):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if self.cfg['model']['type'] == 'gpt2':
            self.lm_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.max_input_len = GPT2_MAXLEN - self.cfg['dev']['max_resp_length']
        elif self.cfg['model']['type'] == 'bloom':
            self.lm_tokenizer = BloomTokenizerFast.from_pretrained('bigscience/bloom')
            self.max_input_len = BLOOM_MAXLEN - self.cfg['dev']['max_resp_length']
        else:
            logger.error('Model type not supported')
            raise NotImplementedError

        self.ctx_start, self.ctx_end = '[context]', '[endofcontext]'
        self.usr, self.sys = '[usr]', '[sys]'
        self.db_start, self.db_end = '[db]', '[endofdb]'
        self.row_start, self.row_end = '[row]', '[endofrow]'
        self.rsp_start, self.rsp_end = '[response]', '[endofresponse]'
        self.dataset = self.cfg['dataset']
        self.special_tokens = [
            self.ctx_start, self.ctx_end,
            self.usr, self.sys,
            self.db_start, self.db_end,
            self.row_start, self.row_end,
            self.rsp_start, self.rsp_end,
        ]
        self.lm_tokenizer.add_tokens(self.special_tokens, special_tokens=True)
        self.total_vocab_size = len(self.lm_tokenizer)
        self.ent2type = read_entities(os.path.join(cfg['datapath'], 'entities.json'))
        vals = self.lm_tokenizer.encode(self.rsp_end)
        assert len(vals) == 1
        self.eos_token_idx = vals[0]

    def fit(self, data):
        return self

    def serialize_context(self, context):
        decor_uttrs = []
        for ii, uttr in enumerate(context):
            prefix = self.usr if ii % 2 == 0 else self.sys
            decor_uttrs.append(f"{prefix} {uttr}")
        sequence = ' '.join(decor_uttrs)
        sequence = self.ctx_start + sequence + " " + self.ctx_end
        return sequence

    def serialize_kb(self, kb, mode='infer'):
        if len(kb) == 0:
            return ''

        if mode != 'infer':
            np.random.shuffle(kb)
        phrases = []
        for entry in kb:
            if self.dataset == 'babi':
                head_attr = 'r_name'
            elif self.dataset == 'bitod':
                head_attr = 'name'
            else:
                raise NotImplementedError

            phrase = self.row_start + " " + entry[head_attr]
            for rel, tar in entry.items():
                if rel == head_attr:
                    continue
                phrase += f" {rel} {tar},"
            assert phrase[-1] == ','
            phrase = phrase[:-1] + " " + self.row_end
            phrases.append(phrase)
        sequence = ''.join(phrases)
        sequence = self.db_start + sequence + self.db_end
        
        return sequence

    def serialize_target(self, target):
        # We are not using response start here. We consider it as part of the prompt itself.
        return " " + target + " " + self.rsp_end

    def transform_entry(self, sample, mode='infer'):
        ctx_sequence = self.serialize_context(sample['context'])
        kb_sequence = self.serialize_kb(sample['kb'], mode)

        input_ids = self.lm_tokenizer.encode(ctx_sequence + kb_sequence + self.rsp_start)
        labels = [-100 for _ in range(len(input_ids))]

        if mode != 'infer':
            tar_sequence = self.serialize_target(sample['output'])
            tar_ids = self.lm_tokenizer.encode(tar_sequence)
            input_ids.extend(tar_ids)
            labels.extend(tar_ids)

        if len(input_ids) > self.max_input_len:
            input_ids = input_ids[:1] + input_ids[1:self.max_input_len - 2]  + input_ids[-1:]
            labels = labels[:1] + labels[1:self.max_input_len - 2]  + labels[-1:]

        ret = {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'position_ids': np.arange(len(input_ids), dtype=np.int64),
            'attention_mask': np.ones(len(input_ids), dtype=np.int64),
            'labels': np.array(labels, dtype=np.int64),
            'mode': mode,
        }
        ret.update(sample)

        return ret

    def get_ov_config(self):
        ret = dict()        
        return ret
    
    @staticmethod
    def collate_fn(batch):
        # This argument only comes from validations
        pad_left = batch[0].get('mode', 'train') == 'infer'
        max_sequence_length = -1
        for entry in batch:
            if max_sequence_length < len(entry['input_ids']):
                max_sequence_length = len(entry['input_ids'])

        assert max_sequence_length > 0
        bs = len(batch)

        input_token_ids = np.zeros((bs,  max_sequence_length), dtype=np.int64)
        attention_mask = np.zeros((bs,  max_sequence_length), dtype=np.int64)
        position_ids = np.zeros((bs,  max_sequence_length), dtype=np.int64)
        labels = np.ones((bs,  max_sequence_length), dtype=np.int64) * (-100)

        for idx, entry in enumerate(batch):
            length = len(entry['input_ids'])
            if pad_left:
                input_token_ids[idx, -length:] = entry['input_ids']
                attention_mask[idx, -length:] = entry['attention_mask']
                position_ids[idx, -length:] = entry['position_ids']
                labels[idx, -length:] = entry['labels']
            else:
                input_token_ids[idx, :length] = entry['input_ids']
                attention_mask[idx, :length] = entry['attention_mask']
                position_ids[idx, :length] = entry['position_ids']
                labels[idx, :length] = entry['labels']

        input_token_ids = torch.tensor(input_token_ids)
        attention_mask = torch.tensor(attention_mask)
        position_ids = torch.tensor(position_ids)
        labels = torch.tensor(labels)

        assert input_token_ids.shape == attention_mask.shape
        assert input_token_ids.shape == position_ids.shape
        assert input_token_ids.shape == labels.shape
        ret = {
            "input_ids": input_token_ids, "position_ids": position_ids,
            "attention_mask": attention_mask, "labels": labels,
        }
        return ret
