import json
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description='Prepare relation extraction dataset'
    )
    parser.add_argument(
        "-src_file",
        "--src_file",
        help="Source txt datafile",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tar_loc",
        "--tar_loc",
        help="Target location where dataset files are created",
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
    tar_loc = args['tar_loc']
    dataset = args['dataset']

    if dataset == 'babi':
        from babi_dlg_utils import read_dialogs
    elif dataset == 'bitod':
        from bitod_dlg_utils import read_dialogs
    else:
        raise NotImplementedError

    dialogs = read_dialogs(src_file)
    gold_data, infer_data = [], []
    for dlg in dialogs:
        gold_samples, infer_samples = dlg.get_samples()
        gold_data.extend(gold_samples)
        infer_data.extend(infer_samples)

    tag = os.path.split(src_file)[-1]
    tag = tag[:tag.index('.')]
    os.makedirs(tar_loc, exist_ok=True)

    print(f'Number of gold samples: {len(gold_data)}')
    fname = os.path.join(tar_loc, f"{tag}_gold.json")
    with open(fname, 'w') as fp:
        json.dump(gold_data, fp, indent=2)

    print(f'Number of infer samples: {len(infer_data)}')
    if len(infer_data) == 0:
        return
    fname = os.path.join(tar_loc, f"{tag}_infer.json")
    with open(fname, 'w') as fp:
        json.dump(infer_data, fp, indent=2)


if __name__ == '__main__':
    args = get_args()
    run(args)
