import os
import json
import numpy as np
import pandas as pd
from copy import deepcopy


class BaseEngine(object):
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
        'available_options',

        # Restaurant
        'address',
        'cuisine',
        'date',
        'dietary_restrictions',
        'number_of_people',

        # Attraction
        'type'
    ]

    def __init__(self, kb_file, supported_api_call):
        self.kb_file = kb_file
        self.kb_df = None
        self.all_columns = None
        self.canon_map = None
        self.supported_api_call = supported_api_call
        # self.cols_to_consider = COLS_TO_CONSIDER
        self.load_kb()

    def load_kb(self):
        with open(self.kb_file, 'r') as fp:
            data = json.load(fp)
        print(f'Read {len(data)} from {self.kb_file}')
        kb_df = pd.DataFrame(data)
        kb_df['is_available'] = pd.Series([True for _ in range(kb_df.shape[0])])
        self.all_columns = list(kb_df)
        self.kb_df = kb_df

        canon_file = './orig_data/en_entity_map.json'
        with open(canon_file, 'r') as fp:
            self.canon_map = json.load(fp)

        # ADHOC additions
        self.canon_map['one0'] = '10'

        print(f'Dataload complete...')
    
    def _preprocess_entry(self, entry, num_matches, constraints):
        tentry = deepcopy(entry)
        tentry['available_options'] = num_matches

        if 'num_of_rooms' in entry:
            tentry['number_of_rooms'] = tentry['num_of_rooms']
            del tentry['num_of_rooms']

        def get_matching_ent(attr, cst_attr):
            assert type(attr) == list
            if len(attr) == 0:
                return None

            if type(cst_attr) == dict:
                return attr[0]

            if type(cst_attr) == str:
                if cst_attr in attr:
                    return cst_attr
                else:
                    return attr[0]

            for val in attr:
                if val in cst_attr:
                    return val
            return attr[0]

        for etype in ['location', 'cuisine', 'dietary_restrictions', 'type']:
            if etype not in entry:
                continue
            if tentry[etype] is None:
                continue

            cst_etypes = [x[0] for x in constraints]
            if etype in cst_etypes:
                cst_attr = constraints[cst_etypes.index(etype)]
                if cst_attr[1] in ['equal_to', 'one_of']:
                    cst_attr = cst_attr[2]
                else:
                    cst_attr = []
                tentry[etype] = get_matching_ent(tentry[etype], cst_attr)
            elif len(tentry[etype]) > 0: 
                tentry[etype] = tentry[etype][0]
            else:
                tentry[etype] = 'None'

        ret = dict()
        punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
        for k in self.cols_to_consider:
            if k not in tentry:
                continue

            text = str(tentry[k]).lower()
            text = text.translate(str.maketrans('', '', punctuation))
            ret[k] = '_'.join(text.split())

        return ret

    def save_kb(self):
        df = self.kb_df
        data = df.to_dict('records')
        results = []
        
        for tentry in data:
            ret = dict()
            punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^`{|}~'
            for k in self.cols_to_consider:
                if k not in tentry:
                    continue

                if type(tentry[k]) == list:
                    ret[k] = []
                    for mm in tentry[k]:
                        text = str(mm).lower()
                        text = text.translate(str.maketrans('', '', punctuation))
                        ret[k].append('_'.join(text.split()))
                else:
                    text = str(tentry[k]).lower()
                    text = text.translate(str.maketrans('', '', punctuation))
                    ret[k] = '_'.join(text.split())

            results.append(ret)

        with open('orig_data/all_kb.json', 'w') as fp:
            json.dump(results, fp, indent=2)

    def _process_constraints(self, dict_data):
        # convert the dictionary in the data to api
        # Directly from BiTOD code...
        def is_int(val):
            try:
                num = int(val)
            except ValueError:
                return False
            return True

        constraints = []
        for const in dict_data:
            for slot, values in const.items():
                relation = values[values.find(".")+1:values.find("(")]
                values = values[values.find("(")+1:-1]

                if relation=="one_of" or relation == 'none_of':
                    # check this
                    values = values.split(" , ")
                    values = [v if type(v) != str else self.canon_map.get(v, v) for v in values]
                else:
                    old = deepcopy(values)
                    values = int(values) if is_int(values) else values
                    if type(values) == str:
                        values = self.canon_map.get(values, old)
                        values = int(values) if is_int(values) else values

                constraints.append((slot, relation, values))

        return constraints

    def _compile_results(self, results, result_size=8):
        def stratified_selection(hotels, num_sel):
            assert len(hotels) > 0, f"Incorrect input to selection."
            buckets = dict()
            for entry in hotels:
                rating = entry['rating']
                if rating not in buckets:
                    buckets[rating] = []
                buckets[rating].append(entry)

            buck_ids = sorted(buckets.keys())[::-1]
            per_bucket = int(np.ceil(num_sel / len(buck_ids)))

            ret = []
            for bid in buck_ids:
                buckets[bid] = sorted(
                    buckets[bid],
                    key=lambda x: (-int(x['rating']), int(x['_id']))
                )
                ssize = min(num_sel - len(ret), per_bucket)
                if ssize == 0:
                    break
                ret.extend(buckets[bid][:ssize])

            return ret

        ret = []
        num_sel = result_size
        if len(results) > 0:
            ret.extend(stratified_selection(results, num_sel))

        return ret

    def _get_query_mask(self, api_call):
        api = api_call['api']
        all_columns = self.all_columns
        kb_df = self.kb_df
        constraints = self._process_constraints(api_call['constraints'])
        msk = pd.Series([True for _ in range(kb_df.shape[0])])
        for col, rel, val in constraints:
            assert col in all_columns, f"Unsupported column."
            kb_col = kb_df[col]

            if col not in ['location', 'dietary_restrictions', 'cuisine', 'type']:
                if rel == "one_of":
                    tmsk = kb_col.apply(lambda x: x in val)
                elif rel == "at_least":
                    tmsk = kb_col >= val
                elif rel == "not":
                    tmsk = kb_col != val
                elif rel == "less_than":
                    tmsk = kb_col < val
                elif rel == "none_of":
                    tmsk = kb_col.apply(lambda x: x not in val)
                elif rel == 'equal_to':
                    tmsk = kb_col == val
                else:
                    print(f"Operation {rel} is not supported.")
                    raise NotImplementedError
            elif type(val) == list:
                if rel == "one_of":
                    tmsk = kb_col.apply(lambda y: any(x in val for x in y) if y is not None else False)
                elif rel == "not":
                    tmsk = kb_col.apply(lambda y: all(x not in val for x in y) if y is not None else False)
                elif rel == 'equal_to':
                    tmsk = kb_col.apply(lambda y: any(x == val for x in y) if y is not None else False)
                else:
                    print(f"Operation {rel} is not supported.")
                    raise NotImplementedError
            else:
                if rel == "one_of":
                    tmsk = kb_col.apply(lambda y: val in y if y is not None else False)
                elif rel == "not":
                    tmsk = kb_col.apply(lambda y: val not in y if y is not None else False)
                elif rel == 'equal_to':
                    tmsk = kb_col.apply(lambda y: val in y if y is not None else False)
                else:
                    print(f"Operation {rel} is not supported.")
                    raise NotImplementedError

            msk &= tmsk
        msk &= kb_df['is_available']

        return msk

    def query(self, api_call):
        assert api_call['api'] == self.supported_api_call, f"API CALL not supported."
        msk = self._get_query_mask(api_call)
        results = self.kb_df[msk].copy().to_dict('records')

        if len(api_call['additional_constraints']) > 0:
            tapi_call = deepcopy(api_call)
            tapi_call['constraints'].extend(api_call['additional_constraints'])
            msk = self._get_query_mask(tapi_call)
            results = self.kb_df[msk].copy().to_dict('records')

        return results

    def postprocess_results(self, results, api_call, result_size=8):
        num_matches = len(results)
        # results = self._compile_results(results, result_size=result_size)
        constraints = self._process_constraints(api_call['constraints'])

        if len(api_call['additional_constraints']) > 0:
            additional_constraints = self._process_constraints(api_call['additional_constraints'])
            constraints.extend(additional_constraints)

        results = [self._preprocess_entry(entry, num_matches, constraints) for entry in results]

        return results


def api_call_to_sign(api_call):
    tag = deepcopy(api_call['api'])

    vals = []
    for cst in api_call['constraints']:
        vals.extend(cst.items())
    for cst in api_call['additional_constraints']:
        vals.extend(cst.items())
    vals = sorted(vals, key=lambda x: x[0])

    tag = tag + '#' + "#".join([f"{k}:{v}" for k, v in vals])

    return tag
