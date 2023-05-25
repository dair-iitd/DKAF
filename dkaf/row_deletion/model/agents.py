from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .commons import HierarchicalAttentionEncoder, create_sequence_mask
from .gcn import GraphReasoning


class EntryRemoverAgent(nn.Module):
    def __init__(self, cfg):
        super(EntryRemoverAgent, self).__init__()

        cfg = deepcopy(cfg)
        cfg['enable_uttr_pos_emb'] = False
        self.cfg = cfg

        emb_init = cfg['embedding']
        self.embedding = nn.Embedding(
            emb_init.shape[0],
            emb_init.shape[1],
            padding_idx=cfg["pad"],
        )
        self.embedding.weight.data.copy_(torch.from_numpy(emb_init))
        self.encoder = HierarchicalAttentionEncoder(cfg)

        self.graph_reasoning = GraphReasoning(cfg)

        insize = 4 * cfg['enc_hid_size']
        self.predictor = nn.Sequential(
            nn.Linear(in_features=insize, out_features=100, bias=True),
            nn.BatchNorm1d(100, momentum=0.9999),
            nn.Tanh(),
            nn.Linear(100, 1, bias=True)
        ) # Always for class 1 (keep)

        self.use_ent_tags = cfg.get('use_ent_tags', False)
        self.sampling_mode = False

    def set_sampling_mode(self, mode):
        self.sampling_mode = mode

    def compute_logits(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        entry_ent_idxs, entry_ent_lens
    ):
        # 1. Encode dialog
        features = self.embedding(tokens)

        if self.use_ent_tags:
            tag_features = self.embedding(token_tags)
            features = features + tag_features

        res = self.encoder(features, token_lens, dlg_lens)
        query = res[-1]

        # 2. Encode KB
        kb_token_features = self.embedding(kb_tokens)
        kb_type_features = self.embedding(kb_types)
        state = self.graph_reasoning(
            query, kb_token_features, kb_type_features,
            kb_lens, adjacency
        )

        hsize = state[1].size(2)
        gidxs = entry_ent_idxs.unsqueeze(2).expand(-1, -1, hsize)
        node_features = torch.gather(state[1], 1, gidxs)
        gmask = create_sequence_mask(entry_ent_lens, dtype=node_features.dtype)
        node_features = node_features * gmask.unsqueeze(2)
        node_features = torch.sum(node_features, dim=1) / entry_ent_lens.unsqueeze(1)

        cls_features = state[0]
        final_features = torch.cat((cls_features, node_features), dim=1)
        logits = self.predictor(final_features)
        logits = torch.cat((-logits, logits), dim=1)

        return logits

    def forward(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        entry_ent_idxs, entry_ent_lens, targets=None, force=False
    ):
        logits = self.compute_logits(
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            entry_ent_idxs, entry_ent_lens
        )

        if force and targets is not None:
            actions = targets
        else:
            if self.training or self.sampling_mode:
                dist = Categorical(logits=logits)
                actions = dist.sample()
            else:
                actions = torch.argmax(logits, dim=1)

        batch_idx = torch.arange(tokens.size(0), dtype=tokens.dtype)
        log_probs = nn.functional.log_softmax(logits, dim=1)
        log_probs = log_probs[batch_idx, actions]

        return actions, log_probs
