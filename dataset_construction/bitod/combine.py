import json
import sys


def load_json(fname):
    print(f'Loading from {fname}')
    with open(fname, 'r') as fp:
        obj = json.load(fp)
    return obj


data1 = load_json(sys.argv[1])
data2 = load_json(sys.argv[2])
data3 = load_json(sys.argv[3])
data = data1 + data2 + data3
print('Final size', len(data))

with open(sys.argv[4], 'w') as fp:
    json.dump(data, fp, indent=2)
