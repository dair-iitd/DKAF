from copy import deepcopy
import torch
import torch.nn as nn
from .commons import create_sequence_mask, HierarchicalAttentionEncoder
from .gcn import GraphReasoning


class ResidualFF(nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_features=insize, out_features=insize, bias=True),
            nn.BatchNorm1d(insize),
            nn.Tanh(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=insize, out_features=insize, bias=True),
            nn.BatchNorm1d(insize),
            nn.Tanh(),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=insize, out_features=outsize, bias=True),
            nn.BatchNorm1d(outsize),
            nn.Tanh(),
        )

    def forward(self, X):
        Z1 = self.layer1(X)
        Z1 = Z1 + X

        Z2 = self.layer2(Z1)
        Z2 = Z2 + Z1

        Z3 = self.layer3(Z2)

        return Z3


class MEMRewardFunction(nn.Module):
    def __init__(self, cfg):
        super(MEMRewardFunction, self).__init__()

        cfg = deepcopy(cfg)
        cfg['enable_uttr_pos_emb'] = True
        self.cfg = cfg

        self.dropout = cfg['dropout'] = cfg.get('dropout', 0.0)
        print(f'Dropout {self.dropout} enabled.')

        emb_init = cfg['embedding']
        embedding = nn.Embedding(
            emb_init.shape[0],
            emb_init.shape[1],
            padding_idx=cfg["pad"],
        )
        embedding.weight.data.copy_(torch.from_numpy(emb_init))
        self.embedding = embedding

        self.encoder = HierarchicalAttentionEncoder(cfg)
        self.graph_reasoning = GraphReasoning(cfg)

        hid_size = cfg['enc_hid_size']
        self.query_transform = ResidualFF(4 * hid_size, 2 * hid_size)

        insize = 6 * hid_size
        self.tok_copier = nn.Sequential(
            nn.Linear(in_features=insize, out_features=200, bias=True),
            nn.LayerNorm(200),
            nn.ReLU(),
            nn.Linear(200, 1, bias=False)
        )

        insize = 2 * hid_size
        self.inter_prob = nn.Sequential(
            nn.Linear(in_features=insize, out_features=100, bias=True),
            nn.LayerNorm(100),
            nn.ReLU(),
            nn.Linear(100, 1, bias=False),
            nn.Sigmoid()
        )
        self.use_ent_tags = cfg.get('use_ent_tags', False)

    def compute_context_copy_distribution(
        self, token_features, token_lens, dlg_lens,
        query, vocab_size, uttr_vocab_indices
    ):
        shape = token_features.size()  # (bs, max_dlg_len, max_seq_len, 2 * hid)

        # 1. Compute mask
        mask1 = create_sequence_mask(
            token_lens, dtype=token_features.dtype
        )  # (bs, max_dlg_len, max_seq_len)
        mask2 = create_sequence_mask(
            dlg_lens, dtype=token_features.dtype
        ).unsqueeze(2).expand(-1, -1, shape[2])

        mask = mask1 * mask2  # (bs, max_dlg_len, max_uttr_len)

        # 2. Compute logits
        expanded_query = query.unsqueeze(1).unsqueeze(1).expand(-1, shape[1], shape[2], -1)
        feats = torch.cat((token_features, expanded_query), dim=3)
        logits = self.tok_copier(feats)
        logits = logits.squeeze(3)

        # 3. Apply mask
        neg_val = -99999.0
        logits = mask * logits + (1.0 - mask) * neg_val

        # 4. Reshape and softmax (bs, tokens)
        logits = logits.view(shape[0], shape[1] * shape[2])
        probs = nn.functional.softmax(logits, dim=-1)

        # 5. Copy probs to local vocabs
        vocab_idx = uttr_vocab_indices.view(shape[0], -1)
        max_vocab_size = torch.max(vocab_size)
        dist_prob = torch.zeros(
            (shape[0], max_vocab_size), dtype=token_features.dtype,
            device=token_features.device
        )  # (bs, max_vocab_len)

        dist_prob = dist_prob.scatter_add_(1, vocab_idx, probs)

        return dist_prob

    def compute_kb_copy_distribution(
        self, kb_features, kb_lens,
        query, vocab_size, kb_vocab_indices
    ):
        shape = kb_features.size()  # (bs, max_kb_len, emb_size)

        # 1. Compute mask
        mask = create_sequence_mask(
            kb_lens, dtype=kb_features.dtype
        )  # (bs, max_kb_len)

        # 2. Compute logits
        expanded_query = query.unsqueeze(1).expand(-1, shape[1], -1)
        feats = torch.cat((kb_features, expanded_query), dim=2)
        logits = self.tok_copier(feats)
        logits = logits.squeeze(2)

        # 3. Apply mask
        neg_val = -99999.0
        logits = mask * logits + (1.0 - mask) * neg_val

        # 4. Reshape and softmax (bs, tokens)
        probs = nn.functional.softmax(logits, dim=-1)

        # 5. Copy probs to local vocabs
        vocab_idx = kb_vocab_indices.view(shape[0], -1)
        max_vocab_size = torch.max(vocab_size)
        dist_prob = torch.zeros(
            (shape[0], max_vocab_size), dtype=kb_features.dtype,
            device=kb_features.device
        )  # (bs, max_vocab_len)
        dist_prob = dist_prob.scatter_add_(1, vocab_idx, probs)

        return dist_prob

    def get_encoder_attention_scores(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        pos_tokens, ent_uttr_idx, ent_uttr_pos,
        uttr_vocab_indices, kb_vocab_indices,
        vocab_size, ent_types, targets 
    ):
        # 1. Encode dialog
        batch_idx = torch.arange(
            start=0, end=tokens.size(0),
            dtype=tokens.dtype, device=tokens.device
        )
        features = self.embedding(tokens)
        if self.use_ent_tags:
            tag_features = self.embedding(token_tags)
            features = features + tag_features

        dlg_pos_features = self.embedding(pos_tokens)

        res = self.encoder(features, token_lens, dlg_lens, dlg_pos_features, return_attn=True)

        return res[-2], res[-1]

    def compute(
        self, tokens, token_tags, token_lens, dlg_lens,
        kb_tokens, kb_types, kb_lens, adjacency,
        pos_tokens, ent_uttr_idx, ent_uttr_pos,
        uttr_vocab_indices, kb_vocab_indices,
        vocab_size, targets 
    ):
        # 1. Encode dialog
        batch_idx = torch.arange(
            start=0, end=tokens.size(0),
            dtype=tokens.dtype, device=tokens.device
        )
        features = self.embedding(tokens)
        if self.use_ent_tags:
            tag_features = self.embedding(token_tags)
            features = features + tag_features

        dlg_pos_features = self.embedding(pos_tokens)

        res = self.encoder(features, token_lens, dlg_lens, dlg_pos_features)
        cls_features = res[-1]
        dlg_src_feats = res[0][batch_idx, ent_uttr_idx, ent_uttr_pos, :]
        query = torch.cat((cls_features, dlg_src_feats), dim=1)
        query = self.query_transform(query)

        # 2. Encode KB
        kb_token_features = self.embedding(kb_tokens)
        kb_type_features = self.embedding(kb_types)
        state = self.graph_reasoning(
            query, kb_token_features, kb_type_features,
            kb_lens, adjacency
        )
        query = torch.cat((state[0], dlg_src_feats), dim=1)  # (bs, esize + 2 * hid)

        # 3. Compute probs
        ctx_tok_features = res[0]
        context_copy_prob = self.compute_context_copy_distribution(
            ctx_tok_features, token_lens, dlg_lens,
            query, vocab_size, uttr_vocab_indices
        )

        node_features = state[1]
        kb_copy_prob = self.compute_kb_copy_distribution(
            node_features, kb_lens,
            query, vocab_size, kb_vocab_indices
        )

        # 4. Compute final prob
        inter_prob = self.inter_prob(state[0])
        copy_prob = inter_prob * context_copy_prob + (1.0 - inter_prob) * kb_copy_prob

        tprobs = copy_prob[batch_idx, targets]
        losses = -1.0 * torch.log(tprobs)
        loss = torch.mean(losses)

        return loss, copy_prob, losses

    def forward(self, *args):
        (
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            pos_tokens, ent_uttr_idx, ent_uttr_pos,
            uttr_vocab_indices, kb_vocab_indices,
            vocab_size, targets 
        ) = args

        return self.compute(
            tokens, token_tags, token_lens, dlg_lens,
            kb_tokens, kb_types, kb_lens, adjacency,
            pos_tokens, ent_uttr_idx, ent_uttr_pos,
            uttr_vocab_indices, kb_vocab_indices,
            vocab_size, targets 
        )
