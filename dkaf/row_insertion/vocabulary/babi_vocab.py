import numpy as np
from copy import deepcopy
from collections import Counter

import torch


class Vocabulary(object):
    def __init__(self):
        super().__init__()

        self.max_pos_cnt = None
        self.max_seq_len = None

        self.spl_vocab_cnt = 0
        self.uttr_vocab_cnt = 0
        self.total_vocab_size = 0

        self.token2idx = None
        self.idx2token = None
        self.unkid = -1
        self.all_relations = None
        self.unk_prob = 0.0
        self.all_target_relations = None

        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.E1S = '<e1>'
        self.E1E = '<\e1>'
        self.E2S = '<e2>'
        self.E2E = '<\e2>'
        self.reserved_tokens = [self.PAD, self.UNK, self.E1S, self.E1E, self.E2S, self.E2E]

    def get_pos_tokens(self, max_pos):
        print(f'Maximum Position Length {max_pos}')

        ret = []
        for idx in range(-max_pos, max_pos, 1):
            ret.append(f"@{idx}")
        ret.append("@unk")

        return ret

    def build_ordered_vocab_from_token_sets(
        self,
        spl_tokens,
        uttr_tokens
    ):
        print('---- Vocabulary Statistics ----')
        print(f'Special tokens: {len(spl_tokens)}')
        print(f'Context only tokens: {len(uttr_tokens)}')
        print('-------------------------------')

        self.spl_vocab_cnt = len(spl_tokens)
        self.uttr_vocab_cnt = len(uttr_tokens)

        all_tokens = [x for x in spl_tokens]
        all_tokens += sorted(list(uttr_tokens))

        self.total_vocab_size = len(all_tokens)
        assert len(all_tokens) == len(set(all_tokens)), Counter(all_tokens).most_common(5)

        return all_tokens

    def fit(self, data):
        """
        param data: [dict] build vocab from the dialog data
        """
        all_text = []
        for entry in data:
            for uttr in entry['context']:
                all_text.append(uttr)

            for uttr in entry['context_tag']:
                all_text.append(uttr)

        max_ctx_len = max([len(e['context']) for e in data])
        self.max_pos_cnt = max_ctx_len - 2

        # Build Vocabulary
        all_uttr_tokens = [x for y in all_text for x in y.split(' ')]
        unq_uttr_tokens = set(all_uttr_tokens)
        print(f'Unique utterance tokens: {len(unq_uttr_tokens)}')

        spl_tokens = self.reserved_tokens
        spl_tokens.extend(self.get_pos_tokens(self.max_pos_cnt))
        all_tokens = self.build_ordered_vocab_from_token_sets(
            spl_tokens, unq_uttr_tokens
        )

        idxs = list(range(self.total_vocab_size))
        self.idx2token = dict(zip(idxs, all_tokens))
        self.token2idx = dict(zip(all_tokens, idxs))
        self.unkid = self.token2idx[self.UNK]
        print(f'Final context vocabulary size: {self.total_vocab_size}')

        self.all_target_relations = sorted(set([x for e in data for x in e['target']]))
        print('Targets', self.all_target_relations)

        return self

    def transform_utterances(self, context, mode='infer'):
        lens = [len(x.split()) for x in context]
        max_seq_len = max(lens)
        ctx_len = len(context)

        token_ids = []
        seqlens = []

        for idx, uttr in enumerate(context):
            tokens = uttr.split()
            postfix = []
            seqlen = len(tokens)

            tokens = [
                tok if (mode == 'infer' or np.random.rand() > self.unk_prob) else self.UNK
                for tok in tokens
            ]
            postfix.extend([
                self.PAD for _ in range(max_seq_len - seqlen)
            ])
            tokens.extend(postfix)

            if len(tokens) != max_seq_len:
                print('Error')
                print(tokens)
                print(len(tokens), max_seq_len)
            assert len(tokens) == max_seq_len

            tids = [self.token2idx.get(tok, self.unkid) for tok in tokens]

            token_ids.append(tids)
            seqlens.append(seqlen)
        
        ret = {
            'tokens': np.array(token_ids),
            'token_lens': np.array(seqlens),
            'dlg_len': ctx_len,
        }

        return ret

    def transform_entry(self, entry, mode='infer'):
        context = deepcopy(entry['context'])
        source_node = entry['ent1']
        target_node = entry['ent2']
        
        src_uttr_idx, src_uttr_pos = -1, -1
        tar_uttr_idx, tar_uttr_pos = -1, -1

        for idx in range(len(context)):
            if (source_node in context[idx]) and (src_uttr_idx == -1):
                context[idx] = context[idx].replace(
                    source_node, f"{self.E1S} {source_node} {self.E1E}"
                )
                src_uttr_idx = idx
                src_uttr_pos = context[idx].split().index(self.E1S)
            
            if (target_node in context[idx]) and (tar_uttr_idx == -1):
                context[idx] = context[idx].replace(
                    target_node, f"{self.E2S} {target_node} {self.E2E}"
                )
                tar_uttr_idx = idx
                tar_uttr_pos = context[idx].split().index(self.E2S)
    
            if src_uttr_idx != -1 and tar_uttr_idx != -1:
                break

        assert src_uttr_pos != -1 and src_uttr_idx != -1
        assert tar_uttr_pos != -1 and tar_uttr_idx != -1

        pos_unk = self.token2idx["@unk"]
        src_pos_tokens = []
        tar_pos_tokens = []
        for ii in range(len(context)):
            idx = f"@{ii - src_uttr_idx}"
            src_pos_tokens.append(self.token2idx.get(idx, pos_unk))

            idx = f"@{ii - tar_uttr_idx}"
            tar_pos_tokens.append(self.token2idx.get(idx, pos_unk))

        rel_tars = [
            1 if rel in entry['target'] else 0
            for rel in self.all_target_relations
        ]

        ret = {
            'src_uttr_idx': src_uttr_idx,
            'src_uttr_pos': src_uttr_pos,
            'tar_uttr_idx': tar_uttr_idx,
            'tar_uttr_pos': tar_uttr_pos,
            'src_pos_tokens': np.array(src_pos_tokens),
            'tar_pos_tokens': np.array(tar_pos_tokens),
            'target_relations': np.array(rel_tars),
        }
        ctx_ret = self.transform_utterances(context)
        ctx_ret.update(ret)

        context_tag = deepcopy(entry['context_tag'])

        # TODO: Make sure this works when both source and target in same utterance.
        idx, pos = src_uttr_idx, src_uttr_pos
        tokens = context_tag[idx].split()
        tokens.insert(pos, 'null_tag')
        tokens.insert(pos + 2, 'null_tag')
        context_tag[idx] = " ".join(tokens)

        idx, pos = tar_uttr_idx, tar_uttr_pos
        tokens = context_tag[idx].split()
        tokens.insert(pos, 'null_tag')
        tokens.insert(pos + 2, 'null_tag')
        context_tag[idx] = " ".join(tokens)

        assert src_uttr_idx != tar_uttr_idx, "Not supported."

        tag_ret = self.transform_utterances(context_tag)
        ctx_ret['token_tags'] = tag_ret['tokens']

        return ctx_ret

    def get_embeddings(self, emb_size):
        emb = np.random.normal(
            0.0, 1e-2, size=(self.total_vocab_size, emb_size)
        )
        emb[self.token2idx[self.PAD], :] = 0.0

        return emb

    def get_ov_config(self, args):
        emb_size = args['emb_size']
        ret = dict()

        ret['pad'] = self.token2idx[self.PAD]
        ret['embedding'] = self.get_embeddings(emb_size)
        ret['num_labels'] = len(self.all_target_relations)

        return ret

    @staticmethod
    def collate_fn(batch):
        max_dlg_len = -1
        max_tokens_len = -1

        for entry in batch:
            if max_dlg_len < entry['tokens'].shape[0]:
                max_dlg_len = entry['tokens'].shape[0]

            if max_tokens_len < entry['tokens'].shape[1]:
                max_tokens_len = entry['tokens'].shape[1]

        assert max_tokens_len > 0  # we donot expect everything to be pad.

        bs = len(batch)
        tokens = np.zeros((bs, max_dlg_len, max_tokens_len), dtype=np.int64)
        token_tags = np.zeros((bs, max_dlg_len, max_tokens_len), dtype=np.int64)
        token_lens = np.ones((bs, max_dlg_len), dtype=np.int64)
        dlg_lens = np.zeros((bs,), dtype=np.int64)
        src_pos_tokens = np.zeros((bs, max_dlg_len), dtype=np.int64)
        tar_pos_tokens = np.zeros((bs, max_dlg_len), dtype=np.int64)

        src_uttr_idx = np.zeros((bs,), dtype=np.int64)
        src_uttr_pos = np.zeros((bs,), dtype=np.int64)
        tar_uttr_idx = np.zeros((bs,), dtype=np.int64)
        tar_uttr_pos = np.zeros((bs,), dtype=np.int64)

        num_rel_types = len(batch[0]['target_relations'])
        targets = np.zeros((bs, num_rel_types), dtype=np.float32)

        for idx, entry in enumerate(batch):
            shape = entry['tokens'].shape
            tokens[idx, :shape[0], :shape[1]] = entry['tokens']
            token_tags[idx, :shape[0], :shape[1]] = entry['token_tags']
            token_lens[idx, :shape[0]] = entry['token_lens']
            dlg_lens[idx] = entry['dlg_len']

            src_pos_tokens[idx, :shape[0]] = entry['src_pos_tokens']
            tar_pos_tokens[idx, :shape[0]] = entry['tar_pos_tokens']

            src_uttr_idx[idx] = entry['src_uttr_idx']
            src_uttr_pos[idx] = entry['src_uttr_pos']
            tar_uttr_idx[idx] = entry['tar_uttr_idx']
            tar_uttr_pos[idx] = entry['tar_uttr_pos']

            targets[idx, :] = entry['target_relations']

        tokens = torch.tensor(tokens)
        token_tags = torch.tensor(token_tags)
        token_lens = torch.tensor(token_lens)
        dlg_lens = torch.tensor(dlg_lens)
        src_pos_tokens = torch.tensor(src_pos_tokens)
        tar_pos_tokens = torch.tensor(tar_pos_tokens)
        src_uttr_idx = torch.tensor(src_uttr_idx)
        src_uttr_pos = torch.tensor(src_uttr_pos)
        tar_uttr_idx = torch.tensor(tar_uttr_idx)
        tar_uttr_pos = torch.tensor(tar_uttr_pos)
        targets = torch.tensor(targets)

        return (
            tokens, token_tags, token_lens, dlg_lens,
            src_pos_tokens, tar_pos_tokens,
            src_uttr_idx, src_uttr_pos,
            tar_uttr_idx, tar_uttr_pos,
            targets
        )
