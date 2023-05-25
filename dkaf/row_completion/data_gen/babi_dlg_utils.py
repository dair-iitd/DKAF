from copy import deepcopy
import re


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

    def convert_kb_lines_to_dict(self):
        kb_items = dict()

        for s, r, t in self.api_results:
            if s not in kb_items:
                entry = dict()
                entry['R_name'] = s
                kb_items[s] = entry
            kb_items[s][r] = t
        
        kb = list(kb_items.values())

        return kb

    def get_samples(self):
        dlg_context = [utr[1] for utr in self.utterances]
        dlg_ent_types = self.get_dialog_entity_types()
        dialog_ent_type_list = [
            " ".join([dlg_ent_types.get(tok, 'null_tag') for tok in uttr.split()])
            for uttr in dlg_context
        ]
        kb = self.convert_kb_lines_to_dict()

        if len(kb) == 0:
            return []

        aug_samples = []
        infer_samples = []
        for ename, etype in dlg_ent_types.items():
            if etype != 'R_name':
                continue

            tkb = []
            is_infer_sample = False
            for ent in kb:
                tent = deepcopy(ent)
                if tent['R_name'] == ename:
                    if 'R_rating' in tent:
                        del tent['R_rating']
                    else:
                        is_infer_sample = True
                tkb.append(tent)

            sample = dict()
            sample['did'] = self.did
            sample['context'] = dlg_context
            sample['context_tag'] = dialog_ent_type_list
            sample['kb'] = tkb
            tars = (ename, 'R_rating', re.findall(r'resto_.*(\d)stars$', ename)[0])
            sample['target'] = tars
            sample['sign'] = f"{self.did}_{'_'.join(tars)}"

            aug_samples.append(deepcopy(sample))
            if is_infer_sample:
                infer_samples.append(deepcopy(sample))

        return aug_samples, infer_samples

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


def read_dialogs(fname):
    did = 0

    cnt = 0
    with open(fname, 'r') as fp:
        data = []
        dialog = Dialog(did)

        for line in fp.readlines():
            completed = dialog.process_line(line)
            cnt += 1

            if completed:
                if len(dialog.api_results) == 0:
                    print(did, cnt)
                    print(dialog.utterances)
                    import sys
                    sys.exit(0)
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
