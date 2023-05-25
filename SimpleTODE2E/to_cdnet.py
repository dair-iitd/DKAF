import argparse
import json
import pandas as pd
from collections import defaultdict
from conversion_utils import read_babi_dialogs, read_bitod_dialogs
from copy import deepcopy


def get_all_entities(fname):
    print(f'Loading KB from {fname}')

    kb_dict = defaultdict(lambda: {})
    with open(fname, "r") as fp:
        for line in fp.readlines():
            tokens = line.split()
            key = tokens[1]
            rel_key = tokens[2]
            value = tokens[3]
            kb_dict[key][rel_key] = value

        for key, value in kb_dict.items():
            value['R_restro'] = key

        kb_df = pd.DataFrame(kb_dict.values())

    all_entities = {}
    columns = [
        'R_cuisine', 'R_location',
        'R_number', 'R_price', 'R_phone',
        'R_address',
    ]

    all_entities['R_name'] = kb_df.R_restro.tolist()
    all_entities_list = kb_df.R_restro.tolist()

    for col in columns:
        ents = kb_df[col].to_list()
        ents = [x.lower() for x in set(ents)]
        all_entities_list.extend(ents)
        all_entities[col.lower()] = ents

    return all_entities, all_entities_list


def read_cli():
    parser = argparse.ArgumentParser(description="To CDNet/SimpleTOD...")
    parser.add_argument(
        "--src_file",
        help="Source file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--tar_file",
        help="Target file for CDNet/SimpleTOD",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="dataset",
        required=True,
        type=str,
        choices=['babi', 'bitod'],
    )
    parser.add_argument(
        "--ent",
        help="Input is KB file and dump is entities.json.",
        required=False,
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    args = vars(parser.parse_args())

    return args


def run(args):
    src_file = args['src_file']
    tar_file = args['tar_file']

    if args['dataset'] == 'babi':
        data = read_babi_dialogs(src_file)
    else:
        data = read_bitod_dialogs(src_file)

    samples = []

    for obj in data:
        samples.extend(obj.to_cdnet_samples())

    with open(tar_file, 'w') as fp:
        json.dump(samples, fp, indent=2)


def run_ent(args):
    if args['dataset'] == 'babi':
        src_file = args['src_file']
        tar_file = args['tar_file']
        all_entities, all_entities_list = get_all_entities(src_file)

        ret = {
            'all_entities': all_entities,
            'all_entities_list': all_entities_list,
        }
        with open(tar_file, 'w') as fp:
            json.dump(ret, fp, indent=2)

    else:
        src_file = args['src_file']
        tar_file = args['tar_file']

        with open(src_file, 'r') as fp:
            obj = json.load(fp)

        ret = dict()
        ret['all_entities'] = deepcopy(obj)
        ret['all_entities_list'] = sorted(set([ee for ents in obj.values() for ee in ents]))

        with open(tar_file, 'w') as fp:
            json.dump(ret, fp, indent=2)


if __name__ == '__main__':
    args = read_cli()

    if args['ent']:
        run_ent(args)
    else:
        run(args)
