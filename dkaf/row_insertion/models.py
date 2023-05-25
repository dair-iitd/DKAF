from copy import deepcopy
import torch
import torch.nn as nn

from commons import HierarchicalAttentionEncoder


class RelationExtractor(nn.Module):
    def __init__(self, cfg):
        super(RelationExtractor, self).__init__()

        cfg = deepcopy(cfg)
        cfg['enable_dual_uttr_pos_emb'] = True
        self.cfg = cfg

        emb_init = cfg['embedding']
        self.embedding = nn.Embedding(
            emb_init.shape[0],
            emb_init.shape[1],
            padding_idx=cfg["pad"],
        )
        self.embedding.weight.data.copy_(torch.from_numpy(emb_init))
        self.encoder = HierarchicalAttentionEncoder(cfg)

        self.num_classes = cfg['num_labels']
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=cfg['enc_hid_size'] * 6,
                out_features=cfg['enc_hid_size'] * 2
            ),
            nn.BatchNorm1d(cfg['enc_hid_size'] * 2),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=cfg['enc_hid_size'] * 2,
                out_features=self.num_classes
            )
        )
        self.use_ent_tags = cfg.get('use_ent_tags', False)

    def compute_loss(self, logits, labels):
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='mean'
        )

        return loss

    def forward(
        self, tokens, token_tags, token_lens, dlg_lens,
        src_pos_tokens, tar_pos_tokens,
        src_uttr_idx, src_uttr_pos,
        tar_uttr_idx, tar_uttr_pos,
        targets
    ):
        batch_idx = torch.arange(start=0, end=tokens.size(0), dtype=tokens.dtype)
        token_features = self.embedding(tokens)

        if self.use_ent_tags:
            token_tag_features = self.embedding(token_tags)
            token_features = token_features + token_tag_features

        src_pos_features = self.embedding(src_pos_tokens)
        tar_pos_features = self.embedding(tar_pos_tokens)
        pos_features = torch.cat((src_pos_features, tar_pos_features), dim=2)
        res = self.encoder(token_features, token_lens, dlg_lens, pos_features)
        cls_features = res[-1]

        src_features = res[0][batch_idx, src_uttr_idx, src_uttr_pos, :]
        tar_features = res[0][batch_idx, tar_uttr_idx, tar_uttr_pos, :]
        cls_features = torch.cat((cls_features, src_features, tar_features), dim=1)

        # Compute logits
        logits = self.classifier(cls_features)
        loss = self.compute_loss(logits, targets)

        return logits, loss  # (bs, cand_size)
