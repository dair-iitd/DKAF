import sys
import json

def load_json(fname):
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


gold_file = sys.argv[1]
infer_file = sys.argv[2]
tar_file = sys.argv[3]
print(tar_file)

gold_data = load_json(gold_file)
infer_data = load_json(infer_file)

for obj in infer_data:
    sign = obj['sign']
    target = None
    for ee in gold_data:
        if ee['sign'] == sign:
            target = ee['target']
            break
    assert target is not None
    obj['target'] = target

with open(tar_file, 'w') as fp:
    json.dump(infer_data, fp, indent=2)
