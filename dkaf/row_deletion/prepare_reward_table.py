import os
import joblib
import numpy as np
import torch

from utils import read_cli, train_file, load_json, get_dataloader
from environment.environments import Environment

args = read_cli()
print(args)

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

# 2. Get dataloaders
data_loc = args['data_loc']
batch_size = args['batch_size']
fname = os.path.join(data_loc, train_file)
train_dataloader = get_dataloader(fname, vocab, 'train', batch_size)

# 3. Get environment
fname = os.path.join(args['data_loc'], train_file)
neural_eval = Environment(
    fname, vocab, mode='infer', batch_size=args['batch_size'],
    reward_model_dir=args['reward_model_loc'],
    device=args['device'], use_log_prob=False
)

signs = []
rewards = np.zeros((len(neural_eval.dataset), 2))
action = 0
num_batches = neural_eval.num_batches
print('Computing rewards for', action)

jdx = 0
for _ in range(num_batches):
    obs = neural_eval.reset()

    if action == 0:
        signs.extend([x['sign'] for x in neural_eval.batch])

    actions = torch.tensor([action for _ in range(obs[0].size(0))])
    _, rwds, _, _ = neural_eval.step_infer(actions, track_all=False)

    rewards[jdx:jdx + len(rwds), action] = rwds.cpu().numpy()
    rewards[jdx:jdx + len(rwds), 1 - action] = -rewards[jdx:jdx + len(rwds), action]
    jdx = jdx + len(rwds)

reward_table = dict()
for idx, sign in enumerate(signs):
    reward_table[sign] = dict()
    rwds = rewards[idx, :]

    for act, rwd in enumerate(rwds):
        reward_table[sign][act] = rwd

sup_acts = []
for sign, rwds in reward_table.items():
    sup_acts.append(np.argmax([rwds[x] for x in range(2)]))
from collections import Counter
print(Counter(sup_acts))

fname = os.path.join(args['data_loc'], f'train_reward_table.pkl')
joblib.dump(reward_table, fname)

print(f'Training reward table saved at {fname}')
