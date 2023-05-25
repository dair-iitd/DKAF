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
        "--train_file",
        help="Train file",
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


class RuleBasedEntryRemoval(object):
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
        context = obj['context']
        context_tag = obj['context_tag']
        all_entities = self.get_entities_in_sample(context, context_tag)
        src_ent = obj['source']
        flag = False
        head_attr = 'name' if self.dataset == 'bitod' else 'R_name'
        for ent in all_entities:
            if ent[1] == src_ent and ent[0] == head_attr:
                flag = True
                break
        return int(flag)


def run(args):
    dataset = args['dataset']
    train_file = args['train_file']
    tar_file = args['tar_file']

    cls = RuleBasedEntryRemoval(dataset)
    train_data = load_json(train_file)
    cls.fit(train_data)
    predictions = [
        cls.predict(sample) for sample in train_data
    ]
    print(Counter(predictions))

    new_data = []
    for ii, obj in enumerate(train_data):
        pred = predictions[ii]
        ret = {
            'did': obj['did'],
            'source': obj['source'],
            'decision': int(pred),
            'sign': obj['sign']
        }
        new_data.append(ret)

    with open(tar_file, 'w') as fp:
        json.dump(new_data, fp, indent=2)


if __name__ == '__main__':
    args = read_cli()
    run(args)
