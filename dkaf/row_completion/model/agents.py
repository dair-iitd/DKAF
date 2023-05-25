from copy import deepcopy
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .commons import HierarchicalAttentionEncoder
from .gcn import GraphReasoning


class LatentLinkerSupervised(nn.Module):
    def __init__(self, cfg):
        super(LatentLinkerSupervised, self).__init__()

        cfg = deepcopy(cfg)
        cfg['enable_uttr_pos_emb'] = True
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

        hid_size = cfg['enc_hid_size']
        self.query_transform = nn.Sequential(
            nn.Linear(in_features=hid_size * 4, out_features=hid_size * 2, bias=True),
            nn.BatchNorm1d(hid_size * 2),
            nn.Tanh()
        )

        insize = 4 * hid_size
        num_latent_entities = cfg['num_latent_entities']
        self.predictor = nn.Sequential(
            nn.Linear(in_features=insize, out_features=100, bias=True),
            # nn.BatchNorm1d(100),
            nn.Tanh(),
            nn.Linear(100, num_latent_entities, bias=True)
        )

        self.xce_loss = nn.CrossEntropyLoss()
        self.use_ent_tags = cfg.get('use_ent_tags', False)

    def compute_loss(self, targets, logits):
        loss = self.xce_loss(logits, targets)

        return loss

    def compute_logits(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        pos_tokens, source_uttr_idx, source_uttr_pos,
        source_idxs, relations
    ):
        # 1. Encode dialog
        batch_idx = torch.arange(start=0, end=tokens.size(0), dtype=tokens.dtype)
        features = self.embedding(tokens)

        if self.use_ent_tags:
            tag_features = self.embedding(token_tags)
            features = features + tag_features

        dlg_pos_features = self.embedding(pos_tokens)

        res = self.encoder(features, token_lens, dlg_lens, dlg_pos_features)
        cls_features = res[-1]
        dlg_src_feats = res[0][batch_idx, source_uttr_idx, source_uttr_pos, :]
        query = torch.cat((cls_features, dlg_src_feats), dim=1)
        query = self.query_transform(query)

        # 2. Encode KB
        kb_token_features = self.embedding(kb_tokens)
        kb_type_features = self.embedding(kb_types)
        state = self.graph_reasoning(
            query, kb_token_features, kb_type_features,
            kb_lens, adjacency
        )

        cls_features = state[0]
        node_features = state[1][batch_idx, source_idxs, :]
        final_features = torch.cat((cls_features, node_features), dim=1)
        logits = self.predictor(final_features)

        return logits

    def forward(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        pos_tokens, source_uttr_idx, source_uttr_pos,
        source_idxs, relations, targets 
    ):
        logits = self.compute_logits(
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            pos_tokens, source_uttr_idx, source_uttr_pos,
            source_idxs, relations 
        )
        loss = self.compute_loss(targets, logits)

        return loss, logits


class LatentLinkerAgent(LatentLinkerSupervised):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.sampling_mode = False

    def set_sampling_mode(self, mode):
        self.sampling_mode = mode

    def forward(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        pos_tokens, source_uttr_idx, source_uttr_pos,
        source_idxs, relations, targets, force=False
    ):
        logits = self.compute_logits(
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            pos_tokens, source_uttr_idx, source_uttr_pos,
            source_idxs, relations 
        )

        if force:
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
