import torch
import os, random
import joblib
import numpy as np
import torch

from utils import (
    read_cli, train_file, dev_file,
    get_dataloader, load_best_model
)
from trainer import Trainer
from model import MEMRewardFunction

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
fname = os.path.join(data_loc, train_file)
train_dataloader = get_dataloader(fname, vocab, 'infer', batch_size)

fname = os.path.join(data_loc, dev_file)
val_dataloader = get_dataloader(fname, vocab, 'infer', batch_size)

# 3. Get Model
args.update(vocab.get_ov_config(args))
model = load_best_model(args, MEMRewardFunction)

trainer = Trainer(
    model, None, 1, vocab,
    train_dataloader, val_dataloader, scheduler=None,
    outdir=args['dest_loc']
)

ret, _ = trainer.evaluate_model(train_dataloader)
print(f'Training Results: {ret}')

ret, preds = trainer.evaluate_model(val_dataloader)
print(f'Validation Results: {ret}')

fname = os.path.join(args['data_loc'], 'reward_model.bin')
torch.save(model, fname)

fname = os.path.join(args['data_loc'], 'vocab.pkl')
joblib.dump(vocab, fname)
