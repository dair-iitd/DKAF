import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import math


def masked_softmax(
    vector: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    memory_efficient: bool = False,
    mask_fill_value: float = -1e32,
) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            msk_vec = vector * mask
            result = f.softmax(msk_vec, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)

        else:
            masked_vector = vector.masked_fill(
                (1 - mask).type(torch.BoolTensor).to(mask.device), mask_fill_value
            )
            result = f.softmax(masked_vector, dim=dim)

    return result


def create_sequence_mask(seqlens, maxlen=None, dtype=torch.bool):
    """
    Create boolean mask which zeros out out-of-range locations.
    :param seqlens: torch.tensor of shape (bs,)
    :param maxlen: int maximum length. Default max(seqlens) is used.
    :param dtype: type of the mask
    :return mask: torch.boolean mask of size (bs, maxlen)
    """

    if maxlen is None:
        maxlen = seqlens.max()

    device = seqlens.device

    row_vector = torch.arange(0, maxlen, 1, device=device)
    matrix = torch.unsqueeze(seqlens, dim=-1)
    mask = row_vector < matrix

    mask = mask.type(dtype)
    mask = mask.to(device)

    return mask


class HierarchicalEncoder(nn.Module):
    def __init__(self, cfg):
        super(HierarchicalEncoder, self).__init__()

        self.cfg = cfg

        emb_size = cfg['emb_size']
        enc_hid_size = cfg['enc_hid_size']
        enc_num_layers = cfg['enc_num_layers']

        enc_drop_out = cfg.get('enc_drop_out', 0.0)
        enc_drop_out = enc_drop_out if enc_num_layers > 1 else 0.0

        # Token uttr encoding
        self.token_encoder = nn.GRU(
            input_size=emb_size, hidden_size=enc_hid_size,
            num_layers=enc_num_layers, dropout=enc_drop_out,
            batch_first=True, bias=True, bidirectional=True
        )

        # Dialog Uttr encoding
        input_size = 2 * enc_hid_size
        hidden_size = 2 * enc_hid_size

        enable_uttr_pos_emb = cfg.get('enable_uttr_pos_emb', False)
        if enable_uttr_pos_emb:
            input_size += emb_size

        enable_dual_uttr_pos_emb = cfg.get('enable_dual_uttr_pos_emb', False)
        if enable_dual_uttr_pos_emb:
            assert enable_uttr_pos_emb == False
            input_size += 2 * emb_size

        self.dlg_encoder = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=enc_num_layers, dropout=enc_drop_out,
            batch_first=True, bias=True, bidirectional=False
        )

    def _compute_per_utterance_features(self, utt_features, utt_lens):
        in_shape = utt_features.size()

        # 1. Flatten
        features = utt_features.view(
            -1, in_shape[-2], in_shape[-1]
        )  # (bs * max_dlg_len, max_utt_len, emb_size)
        seqlens = utt_lens.view(-1)  # (bs * max_dlg_len)

        # 2. Contextualize
        packed_features = pack_padded_sequence(
            input=features, lengths=seqlens.to('cpu'),
            batch_first=True, enforce_sorted=False
        )
        packed_features, hn = self.token_encoder(packed_features)
        features, _ = pad_packed_sequence(
            packed_features, batch_first=True, padding_value=0.0,
        )  # (bs * max_dlg_len, max_utt_len, 2 * enc_hid_size)
        features = features.view(in_shape[0], in_shape[1], in_shape[2], -1)
        hn = hn.view(2, in_shape[0], in_shape[1], -1)

        return features, hn

    def _compute_per_dialog_features(self, agg_utt_features, dlg_lens):
        packed_features = pack_padded_sequence(
            input=agg_utt_features, lengths=dlg_lens.to('cpu'),
            batch_first=True, enforce_sorted=False
        )
        packed_features, hn = self.dlg_encoder(packed_features)
        features, _ = pad_packed_sequence(
            packed_features, batch_first=True, padding_value=0.0,
        )  # (bs, max_dlg_len, 2 * enc_hid_size)

        return features, hn

    def forward(self, token_features, token_lens, dlg_lens, dlg_pos_features=None):
        ctx_token_features, hn =\
            self._compute_per_utterance_features(token_features, token_lens)

        utt_features = torch.cat((hn[0], hn[1]), dim=2)

        if dlg_pos_features is None:
            tfeats = utt_features
        else:
            tfeats = torch.cat((utt_features, dlg_pos_features), dim=2)

        ctx_utt_features, agg_dlg_features = self._compute_per_dialog_features(
            tfeats, dlg_lens
        )

        return (
            ctx_token_features, utt_features,
            ctx_utt_features, agg_dlg_features[0]
        )


class HierarchicalAttentionEncoder(HierarchicalEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)

        insize = cfg['enc_hid_size'] * 2
        hidsize = cfg['enc_hid_size']
        self.token_scorer = nn.Sequential(
            nn.Linear(in_features=insize, out_features=hidsize, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidsize, out_features=1, bias=False)
        )

        self.uttr_scorer = nn.Sequential(
            nn.Linear(in_features=insize, out_features=hidsize, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidsize, out_features=1, bias=False)
        )

        # self.temperature = cfg.get('temperature', 0.001)
        self.temperature = cfg.get('temperature', 1)

    def _compute_uttr_features_from_token_features(self, token_features, token_lens):
        shape = token_features.size()
        reshaped_features = token_features.view(shape[0] * shape[1], shape[2], shape[3])
        token_scores = self.token_scorer(reshaped_features).squeeze(-1) # (bs * max_dlg_len, max_seq_len, H)
        token_scores /= self.temperature

        token_masks = create_sequence_mask(
            token_lens.view(shape[0] * shape[1]), dtype=token_scores.dtype
        ) # token_features (bs * max_dlg_len, max_seq_len)

        token_scores = token_masks * token_scores + (1.0 - token_masks) * (-99999.0)
        token_probs = nn.functional.softmax(token_scores, dim=-1)

        final_features = torch.matmul(token_probs.unsqueeze(1), reshaped_features).squeeze(1)
        final_features = final_features.view(shape[0], shape[1], shape[3])
        token_probs = token_probs.view(shape[0], shape[1], shape[2])

        return final_features, token_probs

    def _compute_dlg_features_from_uttr_features(self, uttr_features, dlg_lens):
        shape = uttr_features.size()
        reshaped_features = uttr_features
        uttr_scores = self.uttr_scorer(reshaped_features).squeeze(-1) # (bs * max_dlg_len, H)
        uttr_scores /= self.temperature

        uttr_masks = create_sequence_mask(
            dlg_lens, dtype=uttr_scores.dtype
        ) # token_features (bs * max_dlg_len, max_seq_len)

        uttr_scores = uttr_masks * uttr_scores + (1.0 - uttr_masks) * (-99999.0)
        uttr_probs = nn.functional.softmax(uttr_scores, dim=-1)
        uttr_probs = uttr_probs.unsqueeze(1)

        final_features = torch.matmul(uttr_probs, reshaped_features).squeeze(1)
        uttr_probs = uttr_probs.view(shape[0], shape[1])

        return final_features, uttr_probs

    def forward(
        self, token_features, token_lens, dlg_lens,
        dlg_pos_features=None, return_attn=False
    ):
        ctx_token_features, hn =\
            self._compute_per_utterance_features(token_features, token_lens)

        # utt_features = torch.cat((hn[0], hn[1]), dim=2)
        ret1 = self._compute_uttr_features_from_token_features(ctx_token_features, token_lens)
        utt_features, token_probs = ret1[0], ret1[1]

        if dlg_pos_features is None:
            tfeats = utt_features
        else:
            tfeats = torch.cat((utt_features, dlg_pos_features), dim=2)

        ctx_utt_features, _ = self._compute_per_dialog_features(
            tfeats, dlg_lens
        )

        ret2 = self._compute_dlg_features_from_uttr_features(ctx_utt_features, dlg_lens)
        agg_dlg_features, uttr_probs = ret2[0], ret2[1]

        ret = (
            ctx_token_features, utt_features,
            ctx_utt_features, agg_dlg_features, # [0]
        )

        if return_attn:
            ret = ret + (token_probs, uttr_probs)

        return ret


class MemoryModule(nn.Module):
    def __init__(self, cfg):
        super(MemoryModule, self).__init__()

        self.cfg = cfg
        self.hops = cfg['mem_hops']

        attn_scorer_list = []
        qsize = cfg['enc_hid_size'] * 2
        memsize = cfg['emb_size']
        attn_size = cfg['attn_size']

        for _ in range(self.hops):
            attn_scorer = nn.Sequential(
                nn.Linear(
                    in_features=qsize + memsize,
                    out_features=qsize, bias=False
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=qsize,
                    out_features=attn_size, bias=False
                ),
                nn.Tanh(),
                nn.Linear(
                    in_features=attn_size,
                    out_features=1, bias=False
                )
            )
            attn_scorer_list.append(attn_scorer)

        self.kbhop_scorer_list = nn.ModuleList(attn_scorer_list)

    def forward(self, query, kb_token_features, kb_type_features, kb_lens):
        memory_contents = torch.mean(kb_token_features, dim=2)
        mem_shape = memory_contents.size(1)
        kb_mask = create_sequence_mask(kb_lens, dtype=torch.BoolTensor)

        for hop in range(self.hops):
            query_expanded = torch.unsqueeze(query, 1).expand(
                -1, mem_shape, -1
            ) # (bs, kblen, 2 * hid_enc_size)
            query_expanded = torch.cat((memory_contents, query_expanded), dim=2)
            beta_logits_khop = self.kbhop_scorer_list[hop](query_expanded).squeeze(-1)

            tdist = masked_softmax(
                vector=beta_logits_khop,
                mask=kb_mask,
                dim=1,
                memory_efficient=True,
            )  # (bs, kblen)
            tvec = torch.bmm(tdist.unsqueeze(1), memory_contents).squeeze(1)
            query = query + tvec

        return tvec
