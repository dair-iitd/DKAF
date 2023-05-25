import os
import json
import random
import argparse
import numpy as np
from utils import read_dialogs
from kb.attraction import AttractionEngine


def read_cli():
    parser = argparse.ArgumentParser(description="Hotel perturbation")
    parser.add_argument(
        "-src_file",
        "--src_file",
        help="Hotel target file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-kb_loc",
        "--kb_loc",
        help="KB location",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tar_loc",
        "--tar_loc",
        help="Hotel target location",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-ksz", "--kbsize", help="KB Size",
        required=False,
        type=int,
        default=8,
    )
    parser.add_argument(
        "-rating_drop",
        "--rating_drop",
        help="Rating Drop",
        required=False,
        type=float,
        default=0.1
    )
    parser.add_argument(
        "-avail_prob",
        "--avail_prob",
        help="Hotel availibility probability",
        required=False,
        type=float,
        default=0.996
    )
    parser.add_argument(
        "--use_latest_kb",
        help="Whether to use recent KB while saving the simulated dialogs. Setting this parameter to True will lead to inconsistent dialogs. Otherwise, dialogs will have their contemporary KB.",
        required=True,
        type=str,
        choices=['True', 'False']
    )
    args = vars(parser.parse_args())
    args['use_latest_kb'] = args['use_latest_kb'] == 'True'

    return args


def run_attraction(args):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    src_file = args['src_file']
    kb_loc = args['kb_loc']
    tar_loc = args['tar_loc']
    avail_prob = args['avail_prob']
    kbsize = args['kbsize']
    rating_drop = args['rating_drop']
    tag = os.path.split(src_file)[-1][:-5]

    data = read_dialogs(src_file)
    data = [obj for obj in data if obj.task == 'attraction']
    engine = AttractionEngine(
        kb_loc=kb_loc, avail_prob=avail_prob,
        kbsize=kbsize, rating_drop=rating_drop
    )

    idxs_to_rewrite = list(range(len(data)))
    while len(idxs_to_rewrite) > 0:
        print(f'Rewritting {len(idxs_to_rewrite)} hotel dialogs...')
        retry_idxs = []
        for idx in idxs_to_rewrite:
            new_kb = engine.query(data[idx].api_call)
            if not data[idx].rewrite(new_kb):
                retry_idxs.append(idx)
        idxs_to_rewrite = retry_idxs

    if args['use_latest_kb']:
        engine.freeze()
        for idx in range(len(data)):
            data[idx].kb = engine.query(data[idx].api_call)

    os.makedirs(tar_loc, exist_ok=True)
    fname = os.path.join(tar_loc, f'{tag}_attraction.json')
    with open(fname, 'w') as fp:
        json.dump([dlg.to_dict() for  dlg in data], fp, indent=2)


if __name__ == '__main__':
    args = read_cli()

    run_attraction(args)
