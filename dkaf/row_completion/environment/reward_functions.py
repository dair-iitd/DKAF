from copy import deepcopy
import numpy as np
import re, os
import joblib
import torch


class NeuralRewardFunction(object):
    reward_model = None

    def __init__(self, model_dir, device='cpu', use_log_prob=False, use_reward_table=None):
        self.use_reward_table = use_reward_table
        if use_reward_table is None:
            self.load_artifacts(model_dir, device, use_log_prob)
            return

        try:
            if use_reward_table == 'train':
                reward_file = os.path.join(model_dir, 'train_reward_table.pkl')
            elif use_reward_table == 'val':
                reward_file = os.path.join(model_dir, 'dev_reward_table.pkl')
            print(f'Loading reward table from {reward_file}')
            self.reward_table = joblib.load(reward_file)
        except:
            print(f'Failed to load reward table. Falling back to reward model')
            self.use_reward_table = None
            self.load_artifacts(model_dir, device, use_log_prob)

    def load_artifacts(self, model_dir, device='cpu', use_log_prob=False):
        model_file = os.path.join(model_dir, 'reward_model.bin')
        print(f'Loading Reward Model from {model_file}')
        if NeuralRewardFunction.reward_model is None:
            NeuralRewardFunction.reward_model = torch.load(model_file)
        self.model = NeuralRewardFunction.reward_model

        vocab_file = os.path.join(model_dir, 'vocab.pkl')
        self.vocab = joblib.load(vocab_file)

        self.collate_fn = self.vocab.collate_fn
        self.device = torch.device('cpu' if device == 'cpu' else 'cuda')
        self.use_log_prob = use_log_prob

        self.model.to(self.device)
        self.model.eval()

    def prepare_sample(self, entry, action):
        nentry = deepcopy(entry)
        src, rel = nentry['source_entity'], nentry['relation_type']
        for idx in range(len(nentry['kb'])):
            if 'R_name' in nentry['kb'][idx]:
                if nentry['kb'][idx]['R_name'] == src:
                    nentry['kb'][idx][rel] = action
            elif 'name' in nentry['kb'][idx]:
                if nentry['kb'][idx]['name'] == src:
                    nentry['kb'][idx][rel] = action

        samples = self.vocab.transform_entry(nentry)

        return samples

    def compute_model_score(self, states, actions):
        data = [
            self.prepare_sample(states[idx], actions[idx])
            for idx in range(len(states))
        ]
        flat_data = [x for y in data for x in y]

        batch = self.collate_fn(flat_data)
        tbatch = [x.to(self.device) for x in batch]

        with torch.no_grad():
            res = self.model(*tbatch)
            scores = res[1].detach().cpu().numpy()
        targets = batch[-1].numpy()

        st = 0
        lhood = np.zeros(len(data))
        for idx in range(len(data)):
            en = st + len(data[idx])

            tars = targets[st:en]
            probs = scores[st:en, :]
            lhood[idx] = 1.0

            for ii, tar in enumerate(tars):
                lhood[idx] *= probs[ii, tar]
            st = en

        return lhood

    def __call__(self, states, actions, mode='infer'):
        if self.use_reward_table is None:
            curr_prob = self.compute_model_score(states, actions)
            rewards = curr_prob

        else:
            rewards = np.array([
                self.reward_table[states[idx]['sign']][actions[idx]]
                for idx in range(len(states))
            ])

        return rewards
