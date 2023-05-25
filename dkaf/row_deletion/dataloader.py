import json
from copy import deepcopy
from torch.utils.data import Dataset


def load_json(fname):
    """
    :param fname: str filename
    """
    with open(fname, 'r') as fp:
        obj = json.load(fp)

    return obj


class BasicDataset(Dataset):
    def __init__(self, data_loc, vocab, mode='infer'):
        """
        :param data_loc: str destination location
        :param vocab: BasicVocabulary object of prebuilt-vocabulary
        """
        self.data_loc = data_loc
        self.mode = mode
        self.vocab = vocab
        self.raw_data = None
        self.data = None

        self.load_data_from_file(data_loc)

    def load_data_from_file(self, fname):
        """
        Load data from json file.
        :param fname: str location of the data file.
        """
        self.raw_data = load_json(fname)
        print(f'Loaded {len(self.raw_data)} samples from {fname}')
        self.process_data()

    def process_data(self):
        print('Processing data...')
        self.data = []
        for entry in self.raw_data:
            res = self.vocab.transform_entry(entry, self.mode)
            new_entry = deepcopy(entry)
            new_entry.update(res)

            self.data.append(new_entry)

        print('Data processing complete...')

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
