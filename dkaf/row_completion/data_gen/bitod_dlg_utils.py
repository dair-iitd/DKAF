from copy import deepcopy
import json
from itertools import product
import numpy as np


class Dialog(object):
    cols_to_consider = [
        # Hotel
        'stars',
        'name',
        'price_level',
        'price_per_night',
        'rating',
        'location',
        'ref_number',
        'phone_number',
        'number_of_rooms',

        # Restaurant
        'address',
        'cuisine',
        'dietary_restrictions',

        # Attraction
        'type'
    ]

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
        context = [x[0] for x in self.utterances]
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
        all_ents = [x[1] for x in self.utterances]

        if len(self.kb) == 0 or self.task != 'hotels':
            return [], []

        assert self.task == 'hotels'
        aug_samples = []
        infer_samples = []
        covered = set()
        for uid, (_, ents) in enumerate(self.utterances):
            if 'name' not in [x[0] for x in ents]:
                continue

            eidx = [ii for ii in range(len(ents)) if ents[ii][0] == 'name'][0]
            ename = ents[eidx][1]

            if ename in covered:
                continue
            covered.add(ename)

            tkb = []
            is_infer_sample = False
            for ent in self.kb:
                tent = deepcopy(ent)
                if tent['name'] == ename:
                    if 'stars' in tent:
                        del tent['stars']
                    else:
                        is_infer_sample = True
                tkb.append(tent)

            sample = dict()
            sample['did'] = self.num_did
            sample['dilog_id'] = self.did
            sample['context'] = context
            sample['context_tag'] = context_tag
            sample['entities'] = all_ents
            sample['kb'] = tkb
            sample['ent'] = (ents[eidx][0], ents[eidx][1], uid, ents[eidx][2])
            tars = (ename, 'stars', '5')
            sample['target'] = tars
            sample['sign'] = f"{self.did}_{'_'.join(tars)}"
            sample['task'] = self.task

            aug_samples.append(deepcopy(sample))
            if is_infer_sample:
                infer_samples.append(deepcopy(sample))

        return aug_samples, infer_samples

    def augment(self, rels):
        if len(rels) == 0:
            return

        enames = [x['name'] for x in self.kb]
        for src, rel, tar in rels:
            idx = enames.index(src)
            self.kb[idx][rel] = tar


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
