import os, random
import joblib
import numpy as np
import torch
import torch.nn as nn

from utils import (
    read_cli, train_file, dev_file, load_json,
    get_dataloader, get_optimizer, get_model
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

if args['dataset'] == 'babi':
    from vocabulary.babi_vocab import Vocabulary
elif args['dataset'] == 'bitod':
    from vocabulary.bitod_vocab import Vocabulary
else:
    raise NotImplementedError


data_loc = args['data_loc']

# 1. Prepare Vocab
fname = os.path.join(data_loc, train_file)
train_data = load_json(fname)
vocab = Vocabulary(use_past_only=args['use_past_only'])
vocab.fit(train_data)
joblib.dump(vocab, os.path.join(args['dest_loc'], 'vocab.pkl'))

# 2. Get dataloaders
data_loc = args['data_loc']
batch_size = args['batch_size']
fname = os.path.join(data_loc, train_file)
train_dataloader = get_dataloader(fname, vocab, 'train', batch_size)

fname = os.path.join(data_loc, dev_file)
val_dataloader = get_dataloader(fname, vocab, 'infer', batch_size)

# 3. Get Model
args.update(vocab.get_ov_config(args))
model = get_model(args, MEMRewardFunction)

# 4. Get optimizer
optimizer = get_optimizer(model, args['learning_rate'])

# 5. Clip Gradient
clip = args['clip']
if clip is not None:
    nn.utils.clip_grad_norm_(model.parameters(), clip)

num_epochs = args['num_epochs']
trainer = Trainer(
    model, optimizer, num_epochs, vocab,
    train_dataloader, val_dataloader, scheduler=None,
    outdir=args['dest_loc']
)
trainer.train()
