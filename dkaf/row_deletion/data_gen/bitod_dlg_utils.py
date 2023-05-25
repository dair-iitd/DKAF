from copy import deepcopy
import json
from itertools import product


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

        kb = self.kb
        if len(kb) == 0:
            return []

        entry_entities = []
        for entry in kb:
            ents = set()
            for attr, val in entry.items():
                ents.add((attr, val))
            entry_entities.append(ents)

        common_entities = deepcopy(entry_entities[0])
        for ents in entry_entities[1:]:
            common_entities = common_entities.intersection(ents)

        dlg_entities =  set()
        for uid in range(1, len(self.utterances), 2):
            for tp, en, _ in self.utterances[uid][1]:
                dlg_entities.add((tp, en))

        entities = [x[1] for x in self.utterances]
        samples = []
        for ii, entry in enumerate(kb):
            ents = deepcopy(entry_entities[ii])
            ents = ents - common_entities

            if len(ents) == 0:
                continue

            if any(e in dlg_entities for e in ents):
                continue
            
            samples.append(dict())
            samples[-1]['did'] = self.num_did
            samples[-1]['context'] = context
            samples[-1]['context_tag'] = context_tag
            samples[-1]['source'] = entry['name']
            samples[-1]['kb'] = deepcopy(kb)
            samples[-1]['sign'] = f"{self.did}_{entry['name']}"
            samples[-1]['entities'] = entities

        return samples

    def augment(self, entries_to_remove):
        new_kb = []
        for entry in self.kb:
            if entry['name'] in entries_to_remove:
                continue
            new_kb.append(deepcopy(entry))
        self.kb = new_kb


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
