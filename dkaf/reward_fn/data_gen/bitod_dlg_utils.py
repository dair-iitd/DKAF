from copy import deepcopy
import json
from itertools import product


class Dialog(object):
    def __init__(self, obj, num_did=None) -> None:
        self.did = obj['did']
        self.num_did = num_did
        self.utterances = obj['utterances']
        self.kb = obj['kb']
        self.api_call = obj['api_call']
        self.report = None
        self.is_consistent = None
        self.old_kb = None
        self.task = obj['task']

    def to_dict(self):
        return {
            'did': self.did,
            'kb': self.kb,
            'utterances': self.utterances,
            'api_call': self.api_call,
            'task': self.task,
        }

    #----------------------------------- For Reward Function Training
    def get_samples(self):
        all_ents = set()
        for entry in self.kb:
            for val in entry.values():
                all_ents.add(val)

        flag = True
        for uid in range(1, len(self.utterances), 2):
            all_ents = all_ents.union(set(self.utterances[uid - 1][0].split()))
            ents = self.utterances[uid][1]
            if any(ee[1] not in all_ents for ee in ents):
                flag = False
                break
            all_ents = all_ents.union(set(self.utterances[uid][0].split()))

        if not flag:
            return None

        context = [x[0] for x in self.utterances]
        entities = [x[1] for x in self.utterances]
        context_tag = []
        for uttr, ents in self.utterances:
            tag_toks = []
            for ii, tok in enumerate(uttr.split()):
                tag = 'null_tag'
                for tp, en, pos in ents:
                    if ii == pos:
                        assert en == tok
                        tag = tp
                        break
                tag_toks.append(tag)
            tag_uttr = ' '.join(tag_toks)
            context_tag.append(tag_uttr)
        kb = self.kb

        ret = {
            'did': self.did,
            'context': context,
            'context_tag': context_tag,
            'kb': kb,
            'sign': self.did,
            'entities': entities
        }

        return ret


def read_dialogs(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    
    data = []
    for idx, dlg in enumerate(obj):
        data.append(Dialog(dlg, num_did=idx))

    print(f'Read {len(data)} dialogs from {fname}.')

    return data


def save_dialogs(data, fname):
    dict_data = [dlg.to_dict() for dlg in data]
    with open(fname, 'w') as fp:
        json.dump(dict_data, fp, indent=2)
