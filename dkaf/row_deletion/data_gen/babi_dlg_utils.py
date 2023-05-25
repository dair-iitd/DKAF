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

        entry_entities = []
        for entry in kb:
            ents = set()
            for val in entry.values():
                ents.add(val)
            entry_entities.append(ents)

        common_entities = deepcopy(entry_entities[0])
        for ents in entry_entities[1:]:
            common_entities = common_entities.intersection(ents)

        dlg_entities = list(dlg_ent_types.keys())

        samples = []
        for ii, entry in enumerate(kb):
            ents = deepcopy(entry_entities[ii])
            ents = ents - common_entities

            if len(ents) == 0:
                continue

            if any(e in dlg_entities for e in ents):
                # This entry includes an exclusive entity. Cannot remove.
                continue

            samples.append(dict())
            samples[-1]['did'] = self.did
            samples[-1]['context'] = dlg_context
            samples[-1]['context_tag'] = dialog_ent_type_list
            samples[-1]['source'] = entry['R_name']
            samples[-1]['kb'] = deepcopy(kb)
            samples[-1]['sign'] = f"{self.did}_{entry['R_name']}"

        return samples

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

    def augment(self, entries_to_remove):
        new_api_results = []
        for s, r, t in self.api_results:
            if s in entries_to_remove:
                continue
            new_api_results.append((s, r, t))

        self.api_results = new_api_results


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
