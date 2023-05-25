import re
import numpy as np
from copy import deepcopy
from collections import Counter

import torch


class Vocabulary(object):
    def __init__(self, use_past_only=False):
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
        self.use_past_only = use_past_only
        self.ent_types = None

        if use_past_only:
            print('Considering only past utterances for MEM')

        print('Unknown prob', self.unk_prob)

        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.mask = '<mask>'
        self.reserved_tokens = [self.PAD, self.UNK, self.mask]

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

        all_kb_tokens = []
        all_relations = []
        for entry in data:
            kb = entry['kb']

            for row in kb:
                all_kb_tokens.extend(row.values())
                all_relations.extend(row.keys())

        self.all_relations = sorted(set(all_relations))
        self.num_attrs = len(self.all_relations)
        self.ent_types = self.all_relations

        all_kb_tokens = all_kb_tokens + self.all_relations
        all_kb_tokens = set(all_kb_tokens)
        print(f"Unique of KB tokens {len(all_kb_tokens)}")

        # Build Vocabulary
        all_uttr_tokens = [x for y in all_text for x in y.split(' ')]
        unq_uttr_tokens = set(all_uttr_tokens)
        print(f'Unique utterance tokens: {len(unq_uttr_tokens)}')

        spl_tokens = self.reserved_tokens
        spl_tokens.extend(self.get_pos_tokens(self.max_pos_cnt))
        all_tokens = self.build_ordered_vocab_from_token_sets(
            spl_tokens, unq_uttr_tokens.union(all_kb_tokens)
        )

        idxs = list(range(self.total_vocab_size))
        self.idx2token = dict(zip(idxs, all_tokens))
        self.token2idx = dict(zip(all_tokens, idxs))
        self.unkid = self.token2idx[self.UNK]
        print(f'Final context vocabulary size: {self.total_vocab_size}')

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

    def prepare_samples(self, entry, src=None, rel=None, mode='infer'):
        context_tag = deepcopy(entry['context_tag'])
        context = entry['context']
        all_entities = entry['entities']
        kb = entry['kb']
        kb_entities = set([v for row in kb for v in row.values()])

        ent_uttr_indices = []
        ent_uttr_positions = []
        entities = []
        entity_types = []
        for idx in range(1, len(context), 2):
            ents = all_entities[idx]
            for tp, en, pos in ents:
                entities.append(en)
                ent_uttr_indices.append(idx)
                ent_uttr_positions.append(pos)
                entity_types.append(tp)

        local_vocab = [tok for uttr in context for tok in uttr.split()]
        local_vocab.extend(kb_entities)
        local_vocab = [self.PAD] + sorted(set(local_vocab))
        max_uttr_len = max([len(uttr.split()) for uttr in context])
        tok_vocab_indices = np.zeros((len(context), max_uttr_len), dtype=np.int64)
        for idx, uttr in enumerate(context):
            for jdx, tok in enumerate(uttr.split()):
                tok_vocab_indices[idx][jdx] = local_vocab.index(tok)

        samples = []
        pos_unk = self.token2idx["@unk"]

        for idx in range(len(entities)):
            ent = entities[idx]

            if (src is not None) and (ent != src):
                continue

            uttr_idx = ent_uttr_indices[idx]
            uttr_pos = ent_uttr_positions[idx]

            if self.use_past_only:
                tcontext = deepcopy(context[:uttr_idx + 1])
                tvocab_indices = np.copy(tok_vocab_indices[:uttr_idx + 1, :])
                tcontext_tag = deepcopy(context_tag[:uttr_idx + 1])
            else:
                tcontext = deepcopy(context)
                tvocab_indices = np.copy(tok_vocab_indices)
                tcontext_tag = deepcopy(context_tag)

            toks = tcontext[uttr_idx].split()
            toks[uttr_pos] = self.mask
            tcontext[uttr_idx] = ' '.join(toks)
            tvocab_indices[uttr_idx][uttr_pos] = 0  # padding out the <mask> before final prob calculations

            ret = self.transform_utterances(tcontext, mode=mode)
            ret['ent_uttr_idx'] = uttr_idx
            ret['ent_uttr_pos'] = uttr_pos
            ret['target'] = local_vocab.index(ent)
            ret['target_entity'] = ent
            ret['target_entity_type'] = entity_types[idx]
            ret['sign'] = f"{entry['did']}_{ent}"
            ret['did'] = entry['did']
            ret['uttr_vocab_indices'] = tvocab_indices[:, :ret['tokens'].shape[1]]
            ret['context'] = tcontext

            pos_tokens = []
            for ii in range(len(tcontext)):
                idx = f"@{ii - uttr_idx}"
                pos_tokens.append(self.token2idx.get(idx, pos_unk))
                pos_tokens[-1] = pos_tokens[-1] if (
                    mode == 'infer' or np.random.rand() > self.unk_prob
                ) else pos_unk

            ret['pos_tokens'] = np.array(pos_tokens)
            ret['local_vocab'] = local_vocab
            ret['vocab_size'] = len(local_vocab)

            tokens = tcontext_tag[uttr_idx].split()
            tokens[uttr_pos] = self.mask
            tcontext_tag[uttr_idx] = " ".join(tokens)
            tag_ret = self.transform_utterances(tcontext_tag)
            ret['token_tags'] = tag_ret['tokens']

            samples.append(ret)

        return samples

    def transform_kb(self, entry, mode='infer'):
        entities = []
        ent_to_type = dict()
        for rec in entry['kb']:
            for k, v in rec.items():
                entities.append(v)
                ent_to_type[v] = k

        entities = sorted(list(set(entities)))

        relations = []
        for rec in entry['kb']:
            sid = entities.index(rec['name'])
            rels = list(rec.keys())

            for rel in rels:
                relid = self.all_relations.index(rel)
                did = entities.index(rec[rel])
                if sid == did:
                    continue
                relations.append([relid, sid, did])
                relations.append([relid + self.num_attrs, did, sid])

        orig_entities = deepcopy(entities)
        entities.append(self.PAD)
        orig_entities.append(self.PAD)

        if len(relations) == 0:
            relations.append([0, 0, 0])
        ent_to_type[self.PAD] = self.PAD

        entry_idxs = [self.token2idx.get(tok, self.unkid) for tok in entities]
        entry_idxs = [
            eid if (mode == 'infer' or np.random.rand() > self.unk_prob) else self.unkid
            for eid in entry_idxs
        ]
        entity_types = [ent_to_type[e] for e in orig_entities]
        entity_types = [self.token2idx[t] for t in entity_types]

        return {
            'kb_tokens': np.array(entry_idxs),
            'kb_types': np.array(entity_types),
            'kb_len': len(entities),
            'kb_relations': np.array(relations),
            'orig_entities': orig_entities,
            'rel_cnt': len(self.all_relations) * 2,  # Donot forget the empty
        }

    def transform_entry(self, entry, mode='infer'):
        samples = self.prepare_samples(entry, mode=mode)
        kb_ret = self.transform_kb(entry)

        if len(samples) == 0:
            return []

        local_vocab = samples[0]['local_vocab']
        kb_vocab_indices = np.zeros((kb_ret['kb_len'],), dtype=np.int64)
        for idx, ent in enumerate(kb_ret['orig_entities']):
            kb_vocab_indices[idx] = local_vocab.index(ent)

        for idx in range(len(samples)):
            samples[idx].update(kb_ret)
            samples[idx]['kb_vocab_indices'] = kb_vocab_indices
            samples[idx]['kb_orig'] = deepcopy(entry['kb'])

        return samples

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

        ret['num_attrs'] = self.num_attrs * 2
        ret['ent_types'] = len(self.ent_types)

        return ret

    @staticmethod
    def collate_fn(batch):
        max_dlg_len = -1
        max_tokens_len = -1
        max_kb_len = -1

        for entry in batch:
            if max_dlg_len < entry['tokens'].shape[0]:
                max_dlg_len = entry['tokens'].shape[0]

            if max_tokens_len < entry['tokens'].shape[1]:
                max_tokens_len = entry['tokens'].shape[1]

            if max_kb_len < entry['kb_tokens'].shape[0]:
                max_kb_len = entry['kb_tokens'].shape[0]

        num_attrs = batch[0]['rel_cnt']

        assert max_tokens_len > 0  # we donot expect everything to be pad.
        assert max_kb_len > 0  # we donot expect everything to be pad.

        # Context
        bs = len(batch)
        tokens = np.zeros((bs, max_dlg_len, max_tokens_len), dtype=np.int64)
        token_tags = np.zeros((bs, max_dlg_len, max_tokens_len), dtype=np.int64)
        token_lens = np.ones((bs, max_dlg_len), dtype=np.int64)
        dlg_lens = np.zeros((bs,), dtype=np.int64)

        # KB
        kb_tokens = np.zeros((bs, max_kb_len), dtype=np.int64)
        kb_types = np.zeros((bs, max_kb_len), dtype=np.int64)
        kb_lens = np.zeros((bs,), dtype=np.int64)

        # Targets
        targets = np.zeros((bs,), dtype=np.int64)

        # Node positions
        ent_uttr_idx = np.zeros((bs,), dtype=np.int64)
        ent_uttr_pos = np.zeros((bs,), dtype=np.int64)

        # Relative positions
        pos_tokens = np.zeros((bs, max_dlg_len), dtype=np.int64)

        # Copy computation
        uttr_vocab_indices = np.zeros((bs, max_dlg_len, max_tokens_len), dtype=np.int64)
        kb_vocab_indices = np.zeros((bs, max_kb_len), dtype=np.int64)
        vocab_size = np.zeros((bs,), dtype=np.int64)

        mat_idxs = []
        mat_vals = []

        for idx, entry in enumerate(batch):
            # Context
            shape = entry['tokens'].shape
            tokens[idx, :shape[0], :shape[1]] = entry['tokens']
            token_tags[idx, :shape[0], :shape[1]] = entry['token_tags']
            token_lens[idx, :shape[0]] = entry['token_lens']
            dlg_lens[idx] = entry['dlg_len']

            # KB
            shape = entry['kb_tokens'].shape
            kb_tokens[idx, :shape[0]] = entry['kb_tokens']
            kb_types[idx, :shape[0]] = entry['kb_types']
            kb_lens[idx] = entry['kb_len']

            mat_idxs.extend([idx, t[0], t[1], t[2]] for t in entry['kb_relations'])
            mat_vals.extend([1.0 for _ in entry['kb_relations']])

            # Target
            targets[idx] = entry['target']

            # Node position
            ent_uttr_idx[idx] = entry['ent_uttr_idx']
            ent_uttr_pos[idx] = entry['ent_uttr_pos']

            # Pos tokens
            shape = entry['pos_tokens'].shape
            pos_tokens[idx, :shape[0]] = entry['pos_tokens']

            # Copy computation
            shape = entry['uttr_vocab_indices'].shape
            uttr_vocab_indices[idx, :shape[0], :shape[1]] = entry['uttr_vocab_indices']
            shape = entry['kb_vocab_indices'].shape
            kb_vocab_indices[idx, :shape[0]] = entry['kb_vocab_indices']
            vocab_size[idx] = entry['vocab_size']

        adjacency = torch.sparse_coo_tensor(
            np.array(mat_idxs).T, mat_vals,
            size=(len(batch), num_attrs, max_kb_len, max_kb_len)
        )

        tokens = torch.tensor(tokens)
        token_tags = torch.tensor(token_tags)
        token_lens = torch.tensor(token_lens)
        dlg_lens = torch.tensor(dlg_lens)
        kb_tokens = torch.tensor(kb_tokens)
        kb_types = torch.tensor(kb_types)
        kb_lens = torch.tensor(kb_lens)
        targets = torch.tensor(targets)
        ent_uttr_idx = torch.tensor(ent_uttr_idx)
        ent_uttr_pos = torch.tensor(ent_uttr_pos)
        pos_tokens = torch.tensor(pos_tokens)
        uttr_vocab_indices = torch.tensor(uttr_vocab_indices)
        kb_vocab_indices = torch.tensor(kb_vocab_indices)
        vocab_size = torch.tensor(vocab_size)

        return (
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            pos_tokens, ent_uttr_idx, ent_uttr_pos,
            uttr_vocab_indices, kb_vocab_indices,
            vocab_size, targets
        )
