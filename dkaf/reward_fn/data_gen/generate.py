import json
import argparse


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
        "-tar_file",
        "--tar_file",
        help="Target file",
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
    tar_file = args['tar_file']
    dataset = args['dataset']

    if dataset == 'babi':
        from babi_dlg_utils import read_dialogs
    elif dataset == 'bitod':
        from bitod_dlg_utils import read_dialogs
    else:
        raise NotImplementedError

    dialogs = read_dialogs(src_file)
    samples = []
    for dlg in dialogs:
        samples.append(dlg.get_samples())
    samples = list(filter(lambda x: x is not None, samples))

    print(f'Number of samples: {len(samples)}')
    with open(tar_file, 'w') as fp:
        json.dump(samples, fp, indent=2)


if __name__ == '__main__':
    args = get_args()
    run(args)
