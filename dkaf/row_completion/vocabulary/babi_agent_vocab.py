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

        print('Unknown prob', self.unk_prob)

        self.PAD = '<pad>'
        self.UNK = '<unk>'
        self.E1S = '<e1>'
        self.E1E = '<\e1>'
        self.reserved_tokens = [self.PAD, self.UNK, self.E1S, self.E1E]
        self.all_latent_entities = [
            '1', '2', '3', '4', '5', '6', '7', '8'
        ]

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

    def transform_context(self, entry, mode='infer'):
        context = deepcopy(entry['context'])
        source_node = entry['target'][0]
        node_uttr_idx = -1
        node_uttr_pos = -1

        for idx in range(len(context)):
            if source_node in context[idx]:
                context[idx] = context[idx].replace(
                    source_node, f"{self.E1S} {source_node} {self.E1E}"
                )
                node_uttr_idx = idx
                node_uttr_pos = context[idx].split().index(self.E1S)
                break
        
        assert node_uttr_pos != -1 and node_uttr_idx != -1

        pos_unk = self.token2idx["@unk"]
        pos_tokens = []
        for ii in range(len(context)):
            idx = f"@{ii - node_uttr_idx}"
            pos_tokens.append(self.token2idx.get(idx, pos_unk))

        ret = {
            'source_uttr_idx': node_uttr_idx,
            'source_uttr_pos': node_uttr_pos,
            'pos_tokens': np.array(pos_tokens),
        }
        ctx_ret = self.transform_utterances(context)
        ctx_ret.update(ret)

        idx, pos = node_uttr_idx, node_uttr_pos
        context_tag = deepcopy(entry['context_tag'])
        tokens = context_tag[idx].split()
        tokens.insert(pos, 'null_tag')
        tokens.insert(pos + 2, 'null_tag')
        context_tag[idx] = " ".join(tokens)

        tag_ret = self.transform_utterances(context_tag)
        ctx_ret['token_tags'] = tag_ret['tokens']

        return ctx_ret

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
            sid = entities.index(rec['R_name'])
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
        src, rel, tar = entry['target']

        return {
            'kb_tokens': np.array(entry_idxs),
            'kb_types': np.array(entity_types),
            'kb_len': len(entities),
            'kb_relations': np.array(relations),
            'orig_entities': orig_entities,
            'rel_cnt': len(self.all_relations) * 2,  # Donot forget the empty
            # Targets are here as well
            'source_idx': entities.index(src),
            'relation': self.all_relations.index(rel),
            'target': self.all_latent_entities.index(tar),
            # Token Targets
            'source_entity': src,
            'relation_type': rel,
            'target_entity': tar
        }

    def transform_entry(self, entry, mode='infer'):
        ctx_ret = self.transform_context(entry, mode)
        ctx_ret.update(self.transform_kb(entry, mode))

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
        ret['num_attrs'] = self.num_attrs * 2
        ret['num_latent_entities'] = len(self.all_latent_entities)

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
        source_idxs = np.zeros((bs,), dtype=np.int64)
        targets = np.zeros((bs,), dtype=np.int64)
        relations = np.zeros((bs,), dtype=np.int64)

        # Node positions
        source_uttr_idx = np.zeros((bs,), dtype=np.int64)
        source_uttr_pos = np.zeros((bs,), dtype=np.int64)

        # Relative positions
        pos_tokens = np.zeros((bs, max_dlg_len), dtype=np.int64)

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
            source_idxs[idx] = entry['source_idx']  # Id of source node in KB
            targets[idx] = entry['target']
            relations[idx] = entry['relation']

            # Node position
            source_uttr_idx[idx] = entry['source_uttr_idx']
            source_uttr_pos[idx] = entry['source_uttr_pos']

            # Pos tokens
            shape = entry['pos_tokens'].shape
            pos_tokens[idx, :shape[0]] = entry['pos_tokens']

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
        source_idxs = torch.tensor(source_idxs)
        targets = torch.tensor(targets)
        relations = torch.tensor(relations)
        source_uttr_idx = torch.tensor(source_uttr_idx)
        source_uttr_pos = torch.tensor(source_uttr_pos)
        pos_tokens = torch.tensor(pos_tokens)

        return (
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            pos_tokens, source_uttr_idx, source_uttr_pos,
            source_idxs, relations, targets 
        )
