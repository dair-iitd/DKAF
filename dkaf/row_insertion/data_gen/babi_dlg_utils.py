import re
from itertools import product


class Dialog(object):
    def __init__(self, did=None):
        """
        Class representing a bAbI dialog
        """
        self.utterances = []
        self.api_call = None
        self.api_results = []
        self.api_calls = []
        self.version = 0
        self.did = did

        self.entities = []

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

        # Now we have pure utterance line
        parts[0] = " ".join(parts[0].split()[1:])

        self.utterances.append(('user', parts[0]))
        self.utterances.append(('bot', parts[1]))

        return False

    def get_dialog_entity_types(self):
        context = [x[1] for x in self.utterances]

        ent_types = dict()
        for idx in range(len(context)):
            uttr = context[idx]
            toks = uttr.split()
            if "api_call" in uttr:
                uttr_ents = toks[1:]
                uttr_ents = toks[1:]
                ent_types[uttr_ents[0]] = 'R_cuisine'
                ent_types[uttr_ents[1]] = 'R_location'
                ent_types[uttr_ents[2]] = 'R_number'
                ent_types[uttr_ents[3]] = 'R_price'
            else:
                uttr_ents = list(filter(lambda x: 'resto_' in x, toks))
                for ent in uttr_ents:
                    if 'address' in ent:
                        ent_types[ent] = 'R_address'
                    elif 'phone' in ent:
                        ent_types[ent] = 'R_phone'
                    else:
                        ent_types[ent] = 'R_name'

        return ent_types

    #----------------------------------- For Reward Function Training
    def get_kb(self):
        kb = dict()
        for ss, rr, tt in self.api_results:
            if ss not in kb:
                kb[ss] = {'R_name': ss}
            kb[ss][rr] = tt
        kb = list(kb.values())

        return kb

    def get_samples(self):
        dlg_ent_types = self.get_dialog_entity_types()

        src_candidates = []
        tar_candidates = []
        for ent, etype in dlg_ent_types.items():
            if etype == 'R_name':
                src_candidates.append(ent)
            else:
                tar_candidates.append(ent)
    
        candidate_relations = product(src_candidates, tar_candidates)

        gold_pairs = []
        infer_pairs = []

        for src, tar in candidate_relations:
            rels = []

            head_present = False
            for ent1, rel, ent2 in self.api_results:
                if ent1 == src:
                    head_present = True

                if (ent1 == src) and (ent2 == tar):
                    rels.append(rel)

            if head_present:
                gold_pairs.append([src, rels, tar])
            else:
                assert len(rels) == 0, f'Infer pair cannot have relations in KB.'
                infer_pairs.append([src, tar])

        context = [x[1] for x in self.utterances]
        context_tag = [
            " ".join([dlg_ent_types.get(tok, 'null_tag') for tok in uttr.split()])
            for uttr in context
        ]

        kb = self.get_kb()
        gold_samples = []
        for ent1, rels, ent2 in gold_pairs:
            ret = {
                'did': self.did,
                'context': context,
                'context_tag': context_tag,
                'ent1': ent1, 'ent2': ent2, 'target': rels,
                'sign': f'{self.did}_{ent1}_{ent2}',
                'kb': kb,
            }
            gold_samples.append(ret)

        infer_samples = []
        for ent1, ent2 in infer_pairs:
            ret = {
                'did': self.did,
                'context': context,
                'context_tag': context_tag,
                'ent1': ent1, 'ent2': ent2, 'target': [],
                'sign': f'{self.did}_{ent1}_{ent2}',
                'kb': kb,
            }
            infer_samples.append(ret)

        return gold_samples, infer_samples

    def to_glmp_lines(self):
        utterances = self.utterances

        lines = []
        idx = 0
        cnt = 1
        flag = False
        while idx < len(utterances):
            uttr1 = utterances[idx][1]
            uttr2 = utterances[idx + 1][1]

            if (not flag) and \
                (idx + 3 < len(utterances)) and \
                ('what do you think of this option:' in uttr2):
                flag = True
                for res in self.api_results:
                    src, rel, dest = res
                    lines.append(f"{cnt} {src} {rel} {dest}\n")
                    cnt += 1

            line = f"{cnt} {uttr1}\t{uttr2}\n"
            lines.append(line)
            cnt += 1
            idx += 2

        return lines

    def augment(self, rels):
        self.api_results.extend(rels)


def read_dialogs(fname):
    did = 0
    with open(fname, 'r') as fp:
        data = []
        dialog = Dialog(did)

        for line in fp.readlines():
            completed = dialog.process_line(line)

            if completed:
                data.append(dialog)
                did += 1
                dialog = Dialog(did)

    return data


def save_dialogs(dialogs, tar_file):
    lines = []
    for dialog in dialogs:
        lines.extend(dialog.to_glmp_lines())
        lines.append('\n')

    with open(tar_file, 'w') as fp:
        fp.writelines(lines)
