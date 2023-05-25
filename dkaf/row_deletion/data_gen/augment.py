from collections import defaultdict
import json
import argparse
import os
from tkinter import dialog


def get_args():
    parser = argparse.ArgumentParser(
        description='Augment Relation Extraction Dataset'
    )
    parser.add_argument(
        "-src_file",
        "--src_file",
        help="Source txt datafile",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-aug_file",
        "--aug_file",
        help="Results of relation extraction",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tar_file",
        "--tar_file",
        help="Destination file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-dataset",
        "--dataset",
        help="Dataset",
        required=True,
        type=str,
        choices=['babi', 'bitod'],
    )
    args = vars(parser.parse_args())

    return args
    

def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)

    return obj


def run(args):
    src_file = args['src_file']
    aug_file = args['aug_file']
    tar_file = args['tar_file']
    dataset = args['dataset']

    if dataset == 'babi':
        from babi_dlg_utils import read_dialogs, save_dialogs
    elif dataset == 'bitod':
        from bitod_dlg_utils import read_dialogs, save_dialogs
    else:
        raise NotImplementedError

    dialogs = read_dialogs(src_file)
    did_to_dialogs = dict()

    for dlg in dialogs:
        did_to_dialogs[dlg.did] = dlg

    with open(aug_file, 'r') as fp:
        aug_data = json.load(fp)

    entries_to_remove = defaultdict(lambda : [])
    for ent in aug_data:
        if ent['decision'] == 0:
            did = ent['did']
            src = ent['source']
            entries_to_remove[did].append(src)

    for ii in range(len(dialogs)):
        if args['dataset'] == 'babi':
            rms = entries_to_remove.get(dialogs[ii].did, [])
        else:
            rms = entries_to_remove.get(dialogs[ii].num_did, [])
        dialogs[ii].augment(rms)

    save_dialogs(dialogs, tar_file)


if __name__ == '__main__':
    args = get_args()
    run(args)
