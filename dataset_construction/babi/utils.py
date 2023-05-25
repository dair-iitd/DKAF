import re
import json

api_call_prior_utterance = 'ok let me look into some options for you'


class bAbIDialog(object):
    def __init__(self, ignore_silence=False):
        """
        Class representing a bAbI dialog
        """
        self.utterances = []
        self.api_call = None
        self.api_results = []
        self.api_calls = []
        self.ignore_silence = ignore_silence
        self.version = 0

        self.entities = []

    def set_entities(self):
        """
        Set of tokens to be considered as entities
        :param entities: [str] list of entity tokens
        """
        all_entities = set()
        for api_call in self.api_calls:
            tokens = api_call.split()
            for tok in tokens:
                if tok == 'api_call':
                    continue
                all_entities.add(tok)

        for res in self.api_results['kb']:
            src, rel, dest = res
            all_entities.add(src)
            all_entities.add(dest)

        for uttr in self.utterances:
            tokens = uttr[1].split()
            for tok in tokens:
                if 'resto_' in tok:
                    all_entities.add(tok)

        self.entities = list(all_entities)

    def to_dict(self):
        return {
            'utterances': self.utterances,
            'api_call': self.api_call,
            'api_results': self.api_results['kb'],
            # 'timestamp': self.timestamp,
        }

    def process_line(self, line):
        """
        Update current dialog with input line
        :param line: str
        :return: bool indicating end of the dialog
        """
        if len(line) == 1:
            return True

        if '\t' not in line:
            # api results
            self.api_results.append(line.split()[1:])

            return False

        parts = line.split('\t')
        parts = [part.strip() for part in parts]
        match = re.search(r'api_call', line)
        if match is not None:
            # We found the api call
            self.api_call = parts[1]
            self.api_calls.append(parts[1])
            # return False

        # Now we have pure utterance line
        parts[0] = " ".join(parts[0].split()[1:])

        self.utterances.append(('user', parts[0]))
        self.utterances.append(('bot', parts[1]))

        return False

    def update(self, utterances, api_results):
        self.utterances = utterances
        self.api_results = api_results

    def dialog_to_cdnet_lines_with_api(self):
        utterances = self.utterances
        lines = []
        idx = 0
        cnt = 1
        pattern = r'what do you think of this option'
        kb_flag = False
        while idx < len(utterances):
            uttr1 = utterances[idx][1]
            uttr2 = utterances[idx + 1][1]

            mts = re.findall(pattern, uttr2)
            if len(mts) > 0 and not kb_flag:
                for res in self.api_results['kb']:
                    src, rel, dest = res
                    lines.append(f"{cnt} {src} {rel} {dest}\n")
                    cnt += 1
                kb_flag = True

            line = f"{cnt} {uttr1}\t{uttr2}\n"
            lines.append(line)
            cnt += 1
            idx += 2

        lines.append("\n")

        return lines


def read_babi_dialogs(fname):
    with open(fname, 'r') as fp:
        data = []
        dialog = bAbIDialog()

        for line in fp.readlines():
            completed = dialog.process_line(line)

            if completed:
                data.append(dialog)
                dialog = bAbIDialog()

    return data


def write_dialogs_data(data, fname):
    print(f'Dumping data to {fname}')
    lines = []
    for dialog in data:
        dialog_lines = dialog.dialog_to_cdnet_lines_with_api()
        lines += dialog_lines

    with open(fname, 'w') as fp:
        fp.writelines(lines)


def parse_results(records):
    """
    Convert KB results to list of tuples
    :param res_df:
    :return: [tuple] list of triplets sorted with rating
    """
    columns = [
        'R_phone',
        'R_cuisine',
        'R_address',
        'R_location',
        'R_number',
        'R_price',
        'R_rating',
    ]

    ret_tuples = []
    for record in records:
        rname = record['R_restro']
        for key in columns:
            ret_tuples.append((rname, key, record[key]))

    return ret_tuples
