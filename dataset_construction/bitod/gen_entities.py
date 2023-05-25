import os
import json
import argparse
import numpy as np
from utils import read_dialogs


def read_cli():
    parser = argparse.ArgumentParser(description="BiTOD entities")
    parser.add_argument(
        "-src_loc",
        "--src_loc",
        help="Data source location",
        required=True,
        type=str,
    )
    args = vars(parser.parse_args())

    return args


def run(args):
    src_loc = args['src_loc']

    data = []
    for tag in ['train', 'val', 'test']:
        fname = os.path.join(src_loc, f'{tag}.json')
        data.extend(read_dialogs(fname))

    entities_list = dict()
    for dlg in data:
        for _, ents in dlg.utterances:
            for tp, en, _ in ents:
                if tp not in entities_list:
                    entities_list[tp] = []
                entities_list[tp].append(en)

        for entry in dlg.kb:
            for k, v in entry.items():
                if k not in entities_list:
                    entities_list[k] = []
                entities_list[k].append(v)

    trn_entities = []
    for k in entities_list:
        entities_list[k] = sorted(set(entities_list[k]))
        trn_entities.extend(entities_list[k])
    trn_entities = set(trn_entities)

    fname = os.path.join(src_loc, 'entities.json')
    with open(fname, 'w') as fp:
        json.dump(entities_list, fp, indent=2)

    fname = os.path.join(src_loc, 'val.json')
    data = read_dialogs(fname)
    entities = set()
    for dlg in data:
        for _, ents in dlg.utterances:
            for tp, en, _ in ents:
                entities.add(en)

        for entry in dlg.kb:
            for k, v in entry.items():
                entities.add(en)

    ovr = entities.intersection(trn_entities)
    print(f'Validation entity converage: {len(ovr) / len(entities)}')

    fname = os.path.join(src_loc, 'test.json')
    data = read_dialogs(fname)
    entities = set()
    for dlg in data:
        for _, ents in dlg.utterances:
            for tp, en, _ in ents:
                entities.add(en)

        for entry in dlg.kb:
            for k, v in entry.items():
                entities.add(en)

    ovr = entities.intersection(trn_entities)
    print(f'Test entity converage: {len(ovr) / len(entities)}')


if __name__ == '__main__':
    args = read_cli()
    run(args)
