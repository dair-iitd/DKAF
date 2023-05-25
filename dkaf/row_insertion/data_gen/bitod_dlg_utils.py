from collections import defaultdict
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

    #----------------------------------- For Reward Function Training
    def get_samples(self):
        dlg_entities = set()
        for _, ents in self.utterances:
            for tp, en, _ in ents:
                dlg_entities.add((tp, en))

        # We need to include type here as entities of differnt types can have same value
        src_candidates = []
        tar_candidates = []
        for etype, ent in dlg_entities:
            if etype == 'name':
                src_candidates.append((etype, ent))
            elif etype in self.cols_to_consider:
                tar_candidates.append((etype, ent))
        candidate_relations = product(src_candidates, tar_candidates)

        api_results = []
        for entry in self.kb:
            src = ('name', entry['name'])
            for rel, en in entry.items():
                if rel == 'name':
                    continue
                tar = (rel, en)
                api_results.append((src, rel, tar))

        gold_pairs = []
        infer_pairs = []

        for src, tar in candidate_relations:
            rels = []
            head_present = False
            for ent1, rel, ent2 in api_results:
                if ent1 == src:
                    head_present = True

                if (ent1 == src) and (ent2 == tar):
                    rels.append(rel)

            if head_present:
                gold_pairs.append([src, rels, tar])
            else:
                assert len(rels) == 0, f'Infer pair cannot have relations in KB.'
                infer_pairs.append([src, tar])

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

        # Get entity posistions
        entity_pos = defaultdict(lambda : [])
        for etype, eval in dlg_entities:
            found_pos = None
            for uid, (_, ents) in enumerate(self.utterances):
                for tp, en, pos in ents:
                    if (tp, en) == (etype, eval):
                        found_pos = (uid, pos)
                        entity_pos[(tp, en)].append(found_pos)

        gold_samples = []
        for ent1, rels, ent2 in gold_pairs:
            min_dist = 1e6
            ent1_pos, ent2_pos = None, None
            for pos1 in entity_pos[ent1]:
                for pos2 in entity_pos[ent2]:
                    dist = abs(pos1[0] - pos2[0])
                    if dist < min_dist:
                        min_dist = dist
                        ent1_pos, ent2_pos = pos1, pos2
            tent1 = [ent1[0], ent1[1], ent1_pos[0], ent1_pos[1]]
            tent2 = [ent2[0], ent2[1], ent2_pos[0], ent2_pos[1]]
            ret = {
                'did': self.did,
                'context': context,
                'context_tag': context_tag,
                'ent1': tent1, 'ent2': tent2,
                'target': rels, 'sign': f'{self.did}_{ent1}_{ent2}',
                'kb': self.kb,
            }
            gold_samples.append(ret)

        infer_samples = []
        for ent1, ent2 in infer_pairs:
            min_dist = 1e6
            ent1_pos, ent2_pos = None, None
            for pos1 in entity_pos[ent1]:
                for pos2 in entity_pos[ent2]:
                    dist = abs(pos1[0] - pos2[0])
                    if dist < min_dist:
                        min_dist = dist
                        ent1_pos, ent2_pos = pos1, pos2
            tent1 = [ent1[0], ent1[1], ent1_pos[0], ent1_pos[1]]
            tent2 = [ent2[0], ent2[1], ent2_pos[0], ent2_pos[1]]
            ret = {
                'did': self.did,
                'context': context,
                'context_tag': context_tag,
                'ent1': tent1, 'ent2': tent2,
                'target': [], 'sign': f'{self.did}_{ent1}_{ent2}',
                'kb': self.kb,
            }
            infer_samples.append(ret)

        return gold_samples, infer_samples

    def augment(self, rels):
        if len(rels) == 0:
            return

        new_entries = dict()
        for s, _, _ in rels:
            assert s[0] == 'name'
            if s[1] not in new_entries:
                new_entries[s[1]] = dict()
                new_entries[s[1]]['name'] = s[1]

        for src, rel, tar in rels:
            if tar[0] != rel:
                continue
            assert tar[0] == rel
            new_entries[src[1]][rel] = tar[1]

        new_kb = list(new_entries.values())
        new_kb.extend(self.kb)
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
