import json
import logging

from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger()


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
        logger.info(f'Loaded {len(self.raw_data)} samples from {fname}')
        self.process_data()

    def process_data(self):
        logger.info('Processing data...')
        self.data = []
        for entry in tqdm(self.raw_data, desc='Setting dataset'):
            new_entry = self.vocab.transform_entry(entry, self.mode)
            self.data.append(new_entry)
        logger.info('Data processing complete...')

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
