import torch
import torch.nn as nn
from .commons import create_sequence_mask, masked_softmax


def graph_norm(adjacency):
    """
    Compute normalized adjacency matrix. Similar to
    https://github.com/scoyer/FG2Seq/blob/master/models/modules.py
    """
    graph = adjacency.to_dense()  # (bs, r, n, n)

    degrees = torch.sum(graph, dim=-1, keepdim=True).clamp(min=1)
    graph = graph / degrees

    return graph


class RelationalMessagePassing(nn.Module):
    """
    Similar to
    https://github.com/scoyer/FG2Seq/blob/master/models/layers.py#L292
    """
    def __init__(self, cfg):
        super(RelationalMessagePassing, self).__init__()

        self.cfg = cfg
        self.num_rels = cfg['num_attrs']
        in_size = cfg['emb_size']
        hid_size = cfg['emb_size']

        self.W_r = nn.Parameter(torch.empty(self.num_rels, in_size, hid_size))
        self.W_0 = nn.Parameter(torch.empty(in_size, hid_size))
        self.activation = nn.ReLU()

        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W_0)

    def forward(self, features, adjacency):
        """
        Similar to FG2Seq original implementation.
        :param features: tensor of shape (bs, num_nodes, in_size)
        :param orig_adjacency:
            sparse tensor of size (bs, num_rel, num_nodes, num_nodes)
        """
        feat_shape = features.size()

        # (bs, num_rel, num_nodes, in_size)
        msg_features = torch.matmul(adjacency, features.unsqueeze(1))

        # (num_rel, bs, num_nodes, in_size)
        msg_features = msg_features.permute(1, 0, 2, 3)
        msg_features = torch.reshape(
            msg_features, (self.num_rels, feat_shape[0] * feat_shape[1], -1)
        )  # (num_rel, bs * num_nodes, in_size)

        features_1 = torch.bmm(msg_features, self.W_r)
        features_1 = torch.sum(
            features_1, dim=0
        ).view(feat_shape[0], feat_shape[1], -1)  # (bs, num_nodes, hid_size)
        features_2 = torch.matmul(
            features, self.W_0
        )  # (bs, num_nodes, hid_size)

        features = self.activation(features_1 + features_2)

        return features


class GraphReasoning(nn.Module):
    def __init__(self, cfg):
        super(GraphReasoning, self).__init__()

        self.cfg = cfg
        self.hops = cfg['mem_hops']
        qsize = cfg['enc_hid_size'] * 2
        memsize = cfg['emb_size']
        attn_size = cfg['attn_size']

        message_passing_layers = []
        attn_scorer_list = []
        for _ in range(self.hops):
            message_passing_layers.append(RelationalMessagePassing(cfg))

        self.query_hop = self.hops
        for _ in range(self.query_hop):
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
        self.message_pasing_layers = nn.ModuleList(message_passing_layers)

    def forward(self, query, kb_token_features, kb_type_features, kb_lens, graph):
        memory_contents = kb_token_features + kb_type_features
        mem_shape = memory_contents.size(1)
        kb_mask = create_sequence_mask(kb_lens, dtype=torch.BoolTensor)

        graph = graph_norm(graph)

        for hop in range(self.hops):
            # 1. Update memory (bs, max_kb_token_len, hid_size)
            memory_contents = self.message_pasing_layers[hop](memory_contents, graph)

        for hop in range(self.hops):
            # 2. Update query # (bs, max_kb_token_len, 2 * hid_enc_size)
            query_expanded = torch.unsqueeze(query, 1).expand(-1, mem_shape, -1)
            query_expanded = torch.cat((memory_contents, query_expanded), dim=2)
            logits_khop = self.kbhop_scorer_list[hop](query_expanded).squeeze(-1)

            tdist = masked_softmax(
                vector=logits_khop,
                mask=kb_mask,
                dim=1,
                memory_efficient=True,
            )
            tvec = torch.bmm(tdist.unsqueeze(1), memory_contents).squeeze(1)
            query = query + tvec

        return query, memory_contents
