from copy import deepcopy
import re
import json


class bAbIDialog(object):
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

    def to_cdnet_samples(self):
        # [
        #     'kb', -x done
        #     'context', -x done
        #     'sketch_tags', -x done
        #     'output', -x done
        #     'sketch_outs', -x done
        #     'type', -x done
        #     'gold_entities', -x done
        #     'did', -x done
        #     'kb_ptr',
        #     'key_ptr'
        # ]
        samples = []        

        entries = dict()
        for src, rel, col in self.api_results:
            if src not in entries:
                entries[src] = dict()
                entries[src]['R_name'] = src
            entries[src][rel] = col
        kb = list(entries.values())

        kb_flag = False
        for ii in range(0, len(self.utterances), 2):
            context = deepcopy([x[1] for x in self.utterances[:ii + 1]])
            output = deepcopy(self.utterances[ii + 1][1])

            if ('what do you think of this option:' in output) and (not kb_flag):
                kb_flag = True

            turn_idx = ii // 2

            gold_entities = []
            if 'api_call' in output:
                gold_entities = output.split()[1:]
                ent_to_type = dict(zip(gold_entities, ['R_cuisine', 'R_location', 'R_number', 'R_price']))
            else:
                gold_entities = list(filter(lambda x: '_' in x, output.split()))
                ent_to_type = dict()
                for ent in gold_entities:
                    if "_address" in ent:
                        ent_to_type[ent] = 'R_address'
                    elif "_phone" in ent:
                        ent_to_type[ent] = 'R_phone'
                    else:
                        ent_to_type[ent] = 'R_name'

            sketch_tags = []
            sketch_outs = []
            etypes = []
            for tok in output.split():
                if tok not in gold_entities:
                    sketch_tags.append('null_tag')
                    sketch_outs.append(tok)
                else:
                    sketch_tags.append(tok)
                    etype = ent_to_type.get(tok, None)
                    if etype is None:
                        ntok = tok.replace('_', ' ')
                        etype = ent_to_type[ntok]
                    sketch_outs.append(f"sketch_{etype}")
                    etypes.append(etype)

            sample = dict()
            sample['context'] = [x.lower() for x in context]
            sample['output'] = output.lower()
            sample['type'] = 'restaurant'
            sample['did'] = self.did
            sample['gold_entities'] = [x.lower() for x in gold_entities]
            sample['sketch_tags'] = [x.lower() for x in sketch_tags]
            sample['sketch_outs'] = [x.lower() for x in sketch_outs]

            if len(kb) == 0:
                sample['key_ptr'] = []
                sample['kb_ptr'] = []
                samples.append(sample)
                continue

            sample['key_ptr'] = []
            sample['kb_ptr'] = []
            samples.append(sample)

            if kb_flag:
                fkb = []
                for entry in kb:
                    ee = dict()
                    for k, v in entry.items():
                        ee[k.lower()] = v.lower()
                    fkb.append(ee)
                sample['kb'] = fkb
            else:
                sample['kb'] = []

        return samples


def read_babi_dialogs(fname):
    did = 0
    with open(fname, 'r') as fp:
        data = []
        dialog = bAbIDialog(did)

        for line in fp.readlines():
            completed = dialog.process_line(line)

            if completed:
                data.append(dialog)
                did += 1
                dialog = bAbIDialog(did)

    return data


class BiTODDialog(object):
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

    def to_cdnet_samples(self):
        # [
        #     'kb', -x done
        #     'context', -x done
        #     'sketch_tags', -x done
        #     'output', -x done
        #     'sketch_outs', -x done
        #     'type', -x done
        #     'gold_entities', -x done
        #     'did', -x done
        #     'kb_ptr',
        #     'key_ptr'
        # ]
        samples = []        

        kb = []
        for entry in self.kb:
            keys = sorted(entry.keys())
            ee = dict()
            for key in keys:
                ee[key] = entry[key]
            kb.append(ee)

        for ii in range(0, len(self.utterances), 2):
            context = deepcopy([x[0] for x in self.utterances[:ii + 1]])
            output = deepcopy(self.utterances[ii + 1][0])
            ents = deepcopy(self.utterances[ii + 1][1])

            sketch_tags = []
            sketch_outs = []
            for ii, tok in enumerate(output.split()):
                tag = 'null_tag'
                out = tok
                for tp, en, pos in ents:
                    if pos == ii:
                        assert en == tok, 'Entity does not match token'
                        tag = tok
                        out = f"sketch_{tp}"
                sketch_tags.append(tag)
                sketch_outs.append(out)
            gold_entities = [x[1] for x in ents]
  
            sample = dict()
            sample['context'] = context
            sample['kb'] = deepcopy(kb)
            sample['output'] = output
            sample['type'] = self.task
            sample['did'] = self.num_did
            sample['dlg_sign'] = self.did
            sample['gold_entities'] = gold_entities
            sample['sketch_tags'] = sketch_tags
            sample['sketch_outs'] = sketch_outs
            sample['key_ptr'] = []
            sample['kb_ptr'] = []
            samples.append(sample)

        return samples


def read_bitod_dialogs(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    
    data = []
    for idx, dlg in enumerate(obj):
        data.append(BiTODDialog(dlg, num_did=idx))

    print(f'Read {len(data)} dialogs from {fname}.')

    return data
