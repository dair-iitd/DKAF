from copy import deepcopy
import json
import os, random
import joblib
import numpy as np
import torch

from utils import (
    read_cli, train_file, dev_file, infer_file,
    get_dataloader, load_best_model
)
from trainer import Trainer
from models import RelationExtractor

args = read_cli()
print(args)
os.makedirs(args['dest_loc'], exist_ok=True)

# 0. Set seed
torch.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])

data_loc = args['data_loc']
vocab = joblib.load(os.path.join(args['dest_loc'], 'vocab.pkl'))

# 2. Get dataloaders
data_loc = args['data_loc']
batch_size = args['batch_size']
fname = os.path.join(data_loc, infer_file)
infer_dataloader = get_dataloader(fname, vocab, 'infer', batch_size)

# 3. Get Model
args.update(vocab.get_ov_config(args))
model = load_best_model(args, RelationExtractor)

trainer = Trainer(
    model, None, 1, vocab,
    None, None, scheduler=None,
    outdir=args['dest_loc']
)

ret, logits_arr, _ = trainer.evaluate_model(infer_dataloader, use_metrics=False)
logits_arr = np.array(logits_arr)
preds_arr = (logits_arr > 0).astype(int)
data = infer_dataloader.dataset.raw_data
new_data = []
for ii in range(len(logits_arr)):
    preds = preds_arr[ii]
    if np.sum(preds) == 0:
        continue

    rels = list(filter(lambda idx: preds[idx] > 0, range(logits_arr.shape[1])))
    rels = [vocab.all_target_relations[rr] for rr in rels]
    entry = deepcopy(data[ii])

    for col in ['context', 'context_tag']:
        del entry[col]
    entry['target'] = rels
    new_data.append(entry)

print(f'Added prediction for {len(new_data)} samples.')
fname = os.path.join(data_loc, f'{infer_file[:-5]}_pred.json')
with open(fname, 'w') as fp:
    json.dump(new_data, fp, indent=2)
