from copy import deepcopy
from collections import Counter
import json


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
        self.same_cnt = 0

        self.is_type1 = 0
        self.is_type2 = 0

    def check_consistency(self, all_kb=None):
        kb_ents = set()
        for entry in self.kb:
            for k, v in entry.items():
                kb_ents.add(v)

        num_uttrs = len(self.utterances)
        dlg_tokens = set(self.utterances[0][0].split())
        flag = True
        report = []
        for ii in range(1, num_uttrs, 2):
            ents = [x for x in self.utterances[ii][1]]
            for tt in ents:
                tp, ent = tt[0], tt[1]
                if (ent not in kb_ents) and (ent not in dlg_tokens):
                    flag = False
                    report.append((ii, ent))
            
            tokens = self.utterances[ii][0].split()
            for tok in tokens:
                dlg_tokens.add(tok)

            if ii + 1 < num_uttrs:
                tokens = self.utterances[ii + 1][0].split()
                for tok in tokens:
                    dlg_tokens.add(tok)

            if not flag:
                break

        self.is_consistent = flag
        self.is_type1 = 1 - int(flag) # Missing entity from KB
        self.report = report
        if all_kb is None:
            return

        if len(self.kb) == 0:
            return

        ent_in_dlg = None
        ratings = []
        for ii in range(1, num_uttrs, 2):
            ents = [x for x in self.utterances[ii][1]]
            for tt in ents:
                tp, ent = tt[0], tt[1]
                if tp == 'name' and ent_in_dlg is None:
                    ent_in_dlg = ent
                if tp == 'rating':
                    ratings.append(int(ent))

        if ent_in_dlg is None:
            return

        rating = max(ratings) if len(ratings) > 0 else None
        for entry in all_kb:
            if entry['name'] == ent_in_dlg and rating is None:
                rating = int(entry['rating'])
        krating = max([int(x.get('rating', -1)) for x in self.kb])

        self.is_consistent = flag & (krating <= rating)
        self.is_type2 = int(krating > rating)  # Higher entity in kb
        self.report = report
    
    def __str__(self) -> str:
        ret = dict()

        ret['did'] = self.did
        ret['utterances'] = [x[0] for x in self.utterances]
        ret['kb'] = self.kb

        if self.is_consistent is not None:
            ret['is_consistent'] = self.is_consistent
            ret['report'] = self.report
        
        return json.dumps(ret, indent=2)

    def to_dict(self, handle_avail_opts=True):
        utterances = []
        for uid in range(len(self.utterances)):
            uttr, tents = self.utterances[uid]

            if handle_avail_opts:
                tokens = uttr.split()
                for jj in range(len(tents)):
                    etype, ent, tt = tents[jj]
                    if etype == 'available_options':
                        tokens[tt] = f"#{ent}" if ent[0] != '#' else ent
                        tents[jj] = (etype, f"#{ent}", tt)
                uttr = ' '.join(tokens)

            tents = list(filter(lambda x: x[0] != 'available_options', tents))
            utterances.append([uttr, tents])

        kb = []
        for entry in self.kb:
            tentry = deepcopy(entry)
            if 'available_options' in tentry:
                del tentry['available_options']
            kb.append(tentry)

        return {
            'did': self.did,
            'kb': kb,
            'utterances': utterances,
            'api_call': self.api_call,
            'task': self.task,
        }

    def _replace_row(self, old_row, new_row):
        new_utterances = []

        for uid, (uttr, ents) in enumerate(self.utterances):
            if uid % 2 == 0:
                new_utterances.append([uttr, ents])
                continue

            new_toks = uttr.split()
            new_ents = set()
            for tp, en, pos in ents:
                if tp in old_row and old_row[tp] == en:
                    new_toks[pos] = new_row[tp]
                    new_ents.add((tp, new_row[tp], pos))
                else:
                    new_ents.add((tp, en, pos))
            new_ents = sorted(new_ents, key=lambda x: x[2])
            new_uttr = ' '.join(new_toks)
            new_utterances.append([new_uttr, new_ents])
            tmp = [tuple(x) for x in new_ents]
            assert len(tmp) == len(set(tmp)), f"{tmp}\n{uttr}\n{new_uttr}"

        kb_etypes = ['name', 'address', 'location']
        for uid in range(0, len(new_utterances), 2):
            etypes = [ee[0] for ee in new_utterances[uid][1]]

            if all(x not in etypes for x in kb_etypes):
                continue

            new_ents = []
            new_toks = new_utterances[uid][0].split()
            for tp, en, pos in new_utterances[uid][1]:
                if tp in kb_etypes and old_row[tp] == en:
                    new_toks[pos] = new_row[tp]
                    new_ents.append([tp, new_row[tp], pos])
                else:
                    new_ents.append([tp, en, pos])
            new_utterances[uid] = [' '.join(new_toks), new_ents]

        return new_utterances

    def rewrite(self, new_kb):
        if len(new_kb) == 0:
            return False

        self.old_kb = deepcopy(self.kb)
        self.kb = deepcopy(new_kb)

        hnames = set()
        for uid, (_, ents) in enumerate(self.utterances):
            for tp, en, _ in ents:
                if tp == 'name':
                    hnames.add(en)
        num_rows_to_replace = len(hnames)
        assert num_rows_to_replace == 1, 'Currently only support one suggestion.'

        old_row = self.old_kb[0]
        new_row = self.kb[0]
        self.utterances = self._replace_row(old_row, new_row)

        return True


def read_dialogs(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    
    data = []
    for idx, dlg in enumerate(obj):
        data.append(Dialog(dlg, num_did=idx))

    print(f'Read {len(data)} dialogs from {fname}.')

    return data


if __name__ == '__main__':
    data = read_dialogs('./orig_data/train.json')
    print(Counter([x.task for x in data]))
    data = list(filter(lambda x: x.task == 'hotels', data))

    for dlg in data:
        new_kb = deepcopy(dlg.kb)
        if not dlg.rewrite(new_kb):
            print(f'Rewrite failed.')

    with open('tmp.json', 'w') as fp:
        json.dump([dlg.to_dict() for  dlg in data], fp, indent=2)
