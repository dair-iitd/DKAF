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
gold_eval = Environment(
    fname, vocab, mode='infer', batch_size=args['batch_size'],
    reward_model_dir=args['reward_model_loc'],
    device=args['device'], use_log_prob=False
)
neural_eval = Environment(
    fname, vocab, mode='infer', batch_size=args['batch_size'],
    reward_model_dir=args['reward_model_loc'],
    device=args['device'], use_log_prob=False
)

signs = []
rewards = np.zeros((len(neural_eval.dataset), 2))
num_batches = neural_eval.num_batches

signs = []
rewards = np.zeros((len(neural_eval.dataset), len(vocab.all_latent_entities)))
gold_rewards = np.zeros((len(neural_eval.dataset), len(vocab.all_latent_entities)))
for action in range(len(vocab.all_latent_entities)):
    num_batches = neural_eval.num_batches
    print('Computing rewards for', action)

    jdx = 0
    for _ in range(num_batches):
        obs = neural_eval.reset()
        _ = gold_eval.reset()

        if action == 0:
            signs.extend([x['sign'] for x in neural_eval.batch])

        actions = torch.tensor([action for _ in range(obs[0].size(0))])
        _, rwds, _, _ = neural_eval.step_infer(actions, track_all=False)
        _, gold_rwds, _, _ = gold_eval.step_infer(actions, track_all=False)

        rewards[jdx:jdx + len(rwds), action] = rwds.cpu().numpy()
        gold_rewards[jdx:jdx + len(rwds), action] = gold_rwds.cpu().numpy()
        jdx = jdx + len(rwds)

for idx in range(len(rewards)):
    mn, mx = np.min(rewards[idx]), np.max(rewards[idx])
    rwds = (rewards[idx] == mx).astype(np.float32)
    rewards[idx] = rwds

reward_table = dict()
for idx, sign in enumerate(signs):
    reward_table[sign] = dict()
    rwds = rewards[idx, :]

    for jdx, rwd in enumerate(rwds):
        act = vocab.all_latent_entities[jdx]
        reward_table[sign][act] = 2 * rwd - 1

fname = os.path.join(args['data_loc'], f'train_reward_table.pkl')
joblib.dump(reward_table, fname)

print(f'Training reward table saved at {fname}')

corr_samples = 0
incorr_samples = []
for idx in range(len(gold_rewards)):
    jdx = np.argmax(rewards[idx, :])
    gold_reward = gold_rewards[idx, jdx]

    if gold_reward > 0:
        corr_samples += 1
    else:
        incorr_samples.append(idx)

print(f'Reward Goodness Score {corr_samples / len(rewards)}')

means = np.mean(rewards, axis=1)
stds = np.std(rewards, axis=1)

print(f'Average reward mean = {round(np.mean(means), 5)} Average reward std = {round(np.mean(stds), 5)}')

# for idx in np.random.randint(len(rewards), size=5):
for idx in np.random.choice(incorr_samples, size=5):
    print(f'SIGN: {gold_eval.dataset[idx]["sign"]}')
    str = ""
    for jdx, rw in enumerate(rewards[idx]):
        str = str + f"{round(rw, 5)}({gold_rewards[idx, jdx]})\t"
    print(str)
