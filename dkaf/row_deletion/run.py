import os, random
import joblib
import numpy as np
import torch
import torch.nn as nn
from model import EntryRemoverAgent

from utils import read_cli, train_file, load_json, get_optimizer, get_model
from environment.environments import Environment, MAPOEnvironment
from trainer import RLTrainer, MAPOTrainer

args = read_cli()
print(args)
os.makedirs(args['dest_loc'], exist_ok=True)

# 0. Set seed
torch.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])

if args['dataset'] == 'babi':
    from vocabulary.babi_agent_vocab import Vocabulary
elif args['dataset'] == 'bitod':
    from vocabulary.bitod_agent_vocab import Vocabulary
else:
    raise NotImplementedError


data_loc = args['data_loc']

# 1. Prepare Vocab
fname = os.path.join(data_loc, train_file)
train_data = load_json(fname)
vocab = Vocabulary()
vocab.fit(train_data)
joblib.dump(vocab, os.path.join(args['dest_loc'], 'vocab.pkl'))

# 2. Get dataloaders
data_loc = args['data_loc']
batch_size = args['batch_size']
train_fname = os.path.join(data_loc, train_file)
val_fname = os.path.join(data_loc, train_file)

# 3. Get Environment
train_fname = os.path.join(args['data_loc'], train_file)
val_fname = os.path.join(args['data_loc'], train_file)
if args["use_mapo"]:
    env = MAPOEnvironment(
        train_fname, vocab, mode='train',
        batch_size=args['batch_size'], device=args['device'],
        reward_model_dir=args['reward_model_loc'],
        buff_clip=args['buff_clip'], sample_size=args['sample_size'],
        use_reward_table='train'
    )
else:
    env = Environment(
        train_fname, vocab, mode='train', batch_size=args['batch_size'],
        reward_model_dir=args['reward_model_loc'],
        device=args['device'], use_reward_table='train'
    )

eval_env = Environment(
    val_fname, vocab, mode='infer', batch_size=args['batch_size'],
    reward_model_dir=args['reward_model_loc'],
    device=args['device'], use_reward_table='train'
)

# 4. Get optimizer
args.update(vocab.get_ov_config(args))
model = get_model(args, EntryRemoverAgent)
optimizer = get_optimizer(model, args['learning_rate'])
sched = None

# 5. Clip Gradient
clip = args['clip']
if clip is not None:
    print('Using gradient clipping:', clip)
    nn.utils.clip_grad_norm_(model.parameters(), clip)

# 4. Trainer
num_epochs = args['num_epochs']

if args['use_mapo']:
    trainer = MAPOTrainer(
        model, optimizer, env, eval_env,
        num_epochs=num_epochs, device=args['device'],
        sample_size=args['sample_size'],
        outdir=args['dest_loc'],
        mapo_update_freq=args['mapo_update_freq']
    )

else:
    trainer = RLTrainer(
        model, optimizer, env, eval_env,
        num_epochs=num_epochs, device=args['device'],
        sample_size=args['sample_size'],
        outdir=args['dest_loc'],
        off_sample_prob=args['off_policy_prob']
    )
trainer.train()
