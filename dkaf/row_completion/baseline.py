import re
import json
import argparse
from utils import load_json
from collections import Counter


def read_cli():
    parser = argparse.ArgumentParser(description='Rule based')
    parser.add_argument(
        "-dataset",
        "--dataset",
        help="Dataset to train on",
        required=True,
        type=str,
        choices=['babi', 'bitod']
    )
    parser.add_argument(
        "-tf",
        "--train_infer_file",
        help="Train infer file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-ttf",
        "--tar_file",
        help="Target file",
        required=True,
        type=str,
    )

    args = vars(parser.parse_args())

    return args


class RuleBasedLatentLinking(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def fit(self, data):
        pass

    def get_entities_in_sample(self, context, context_tag):
        entities = []

        for ii, uttr in enumerate(context):
            tokens = uttr.split()
            tag_tokens = context_tag[ii].split()
            for jj, tok in enumerate(tokens):
                tag = tag_tokens[jj]

                if tag != 'null_tag':
                    entities.append([tag, tok, ii, jj])
        
        return entities

    def predict(self, obj):
        if self.dataset == 'babi':
            regex = r"(\d)stars"
            ent = obj['target'][0]
            rating = int(re.findall(regex, ent)[0])

            return rating
        else:
            raise NotImplementedError


def run(args):
    dataset = args['dataset']
    train_infer_file = args['train_infer_file']
    tar_file = args['tar_file']

    cls = RuleBasedLatentLinking(dataset)
    infer_data = load_json(train_infer_file)
    cls.fit(infer_data)
    predictions = [
        cls.predict(sample) for sample in infer_data
    ]
    print(Counter(predictions))

    new_data = []
    for ii, obj in enumerate(infer_data):
        pred = predictions[ii]
        tt = obj['target']
        ret = {
            'did': obj['did'],
            'target': [tt[0], tt[1], pred],
            'sign': obj['sign']
        }
        new_data.append(ret)

    with open(tar_file, 'w') as fp:
        json.dump(new_data, fp, indent=2)


if __name__ == '__main__':
    args = read_cli()
    run(args)
