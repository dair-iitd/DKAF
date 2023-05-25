import os
import json
import argparse

import torch
from dataloader import BasicDataset
from torch.utils.data import DataLoader


train_file = 'train_rd.json'


def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)

    return obj


def read_cli():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument(
        "-dataset",
        "--dataset",
        help="Dataset to train on",
        required=True,
        type=str,
        choices=['babi', 'bitod']
    )
    parser.add_argument(
        "-esize",
        "--emb_size",
        help="Embedding Size",
        required=False,
        type=int,
        default=200,
    )
    parser.add_argument(
        "-ehd",
        "--enc_hid_size",
        help="Encoder Hidden Size",
        required=False,
        type=int,
        default=100,
    )
    parser.add_argument(
        "-enl",
        "--enc_num_layers",
        help="Number of Encoder Layers",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "-bsz",
        "--batch_size",
        help="Batch Size",
        required=False,
        type=int,
        default=32,
    )
    parser.add_argument(
        "-epochs",
        "--num_epochs",
        help="Number of training epochs",
        required=False,
        type=int,
        default=30,
    )
    parser.add_argument(
        "-attn_size",
        "--attn_size",
        help="Attention size",
        required=False,
        type=int,
        default=32,
    )
    parser.add_argument(
        "-mem_hops",
        "--mem_hops",
        help="Number of memory hops",
        required=False,
        type=int,
        default=3,
    )
    parser.add_argument(
        "-clip",
        "--clip",
        help="Clip value",
        required=False,
        type=float,
        default=-1,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning Rate",
        required=False,
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "-dr",
        "--dropout",
        help="dropout",
        required=False,
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "-use_ent_tags",
        "--use_ent_tags",
        help="Use entity type tags",
        required=False,
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "-use_past_only",
        "--use_past_only",
        help="Use past only utterances",
        required=False,
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "-data",
        "--data_loc",
        help="Dataset Location",
        required=True,
        type=str,
        default='./data/',
    )
    parser.add_argument(
        "-tag",
        "--tag",
        help="Model tag to load",
        required=False,
        type=str,
        default="",
    )
    parser.add_argument(
        "-device",
        "--device",
        help="Device to use for training",
        required=False,
        type=str,
        default='cpu',
    )
    parser.add_argument(
        "-dest",
        "--dest_loc",
        help="Destination Location",
        required=True,
        type=str,
        default='./data/',
    )
    parser.add_argument(
        "-seed",
        "--seed",
        help="Seed",
        required=False,
        type=int,
        default=42,
    )
    parser.add_argument(
        "-rw_loc",
        "--reward_model_loc",
        help="Destination Location",
        required=True,
        type=str,
        default='',
    )
    # ---------------- RL arguments ---------------- #
    parser.add_argument(
        "-use_mapo",
        "--use_mapo",
        help="Use MAPO",
        required=False,
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "-sample_size",
        "--sample_size",
        help="Sample Size",
        required=False,
        type=int,
        default=32,
    )
    parser.add_argument(
        "-mapo_update_freq",
        "--mapo_update_freq",
        help="MAPO update frequency",
        required=False,
        type=int,
        default=4,
    )
    parser.add_argument(
        "-off_policy_prob",
        "--off_policy_prob",
        help="Off policy probability",
        required=False,
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "-buff_clip",
        "--buff_clip",
        help="MAPO buffer clip",
        required=False,
        type=float,
        default=0.2,
    )
    args = vars(parser.parse_args())

    return args


def get_dataloader(fname, vocab, mode, batch_size=1):
    dataset = BasicDataset(fname, vocab, mode=mode)
    shuffle = mode == 'train'
    collate_fn = vocab.collate_fn
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collate_fn, drop_last=shuffle
    )

    return dataloader


def get_model(args, Model):
    model = Model(args)

    if args["tag"] == "":
        return model

    fname = os.path.join(args['dest_loc'], args['tag'] + '.bin')
    print('Loading mode from', fname)
    model_weights = torch.load(fname)[1]
    model.load_state_dict(model_weights)

    return model


def get_optimizer(model, lr):
    from torch.optim import Adam
    optimizer = Adam(model.parameters(), lr=lr)

    return optimizer


def load_best_model(args, Model):
    from tinydb import TinyDB
    db = TinyDB(os.path.join(args['dest_loc'], 'logs.db'))
    data = db.all()
    best = max(filter(lambda x: x['epoch'] >= 10, data), key=lambda x: x['average_reward'])
    # best = max(data, key=lambda x: x['average_reward'])

    print(f'Best configuraion: {json.dumps(best, indent=2)}')

    model = Model(args)
    fname = os.path.join(args['dest_loc'], f'model_{best["epoch"]}.bin')
    print('Loading model from', fname)
    model_weights = torch.load(fname)[1]
    model.load_state_dict(model_weights)

    for ent in data:
        if ent['epoch'] == best['epoch']:
            continue
        fname = os.path.join(args['dest_loc'], f'model_{ent["epoch"]}.bin')
        if os.path.exists(fname):
            os.remove(fname)

    return model
