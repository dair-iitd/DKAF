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
gold_eval = Environment(fname, vocab, mode='infer', reward_fn='gold', batch_size=args['batch_size'],)
neural_eval = Environment(
    fname, vocab, mode='infer', batch_size=args['batch_size'],
    reward_fn='neural', reward_model_dir=args['reward_model_loc'],
    device=args['device'], use_log_prob=False
)

signs = []
rewards1 = np.zeros((len(gold_eval.dataset), 2))
rewards2 = np.zeros((len(gold_eval.dataset), 2))

for action in range(2):
    num_batches = gold_eval.num_batches
    print('Computing rewards for', action)

    jdx = 0
    for _ in range(num_batches):
        obs = gold_eval.reset()
        _ = neural_eval.reset()
        signs.extend([x['sign'] for x in gold_eval.batch])

        actions = torch.tensor([action for _ in range(obs[0].size(0))])
        _, rwds1, _, _ = gold_eval.step_infer(actions, track_all=False)
        _, rwds2, _, _ = neural_eval.step_infer(actions, track_all=False)

        rewards1[jdx:jdx + len(rwds1), action] = rwds1.cpu().numpy()
        rewards2[jdx:jdx + len(rwds2), action] = rwds2.cpu().numpy()
        jdx = jdx + len(rwds1)

corr_samples = 0
incorr_samples = []
for idx in range(len(rewards1)):
    jdx = np.argmax(rewards2[idx, :])
    gold_reward = rewards1[idx, jdx]

    if gold_reward > 0:
        corr_samples += 1
    else:
        incorr_samples.append(idx)

print(f'Reward Goodness Score {corr_samples / len(rewards1)}')

tars = np.argmax(rewards1, axis=1)
preds = np.argmax(rewards2, axis=1)
from sklearn.metrics import classification_report
print(classification_report(tars, preds))

from collections import Counter
print(Counter(preds))

means = np.mean(rewards2, axis=1)
stds = np.std(rewards2, axis=1)

print(f'Average reward mean = {round(np.mean(means), 5)} Average reward std = {round(np.mean(stds), 5)}')

# for idx in np.random.randint(len(rewards2), size=5):
for idx in np.random.choice(incorr_samples, size=5):
    print(f'SIGN: {gold_eval.dataset[idx]["did"]}')
    str = ""
    for jdx, rw in enumerate(rewards2[idx]):
        str = str + f"{round(rw, 5)}({rewards1[idx, jdx]})\t"
    print(str)
