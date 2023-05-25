from collections import defaultdict
from copy import deepcopy
import json
import numpy as np
import random
import torch

from .reward_functions import NeuralRewardFunction
from dataloader import BasicDataset

dids_to_track = []


class Environment(object):
    def __init__(
        self, fname, vocab, mode='train', batch_size=1, track=False,
        reward_model_dir='', device='cpu',
        use_log_prob=False, use_reward_table=None
    ):
        self.dataset = BasicDataset(fname, vocab, mode=mode)
        self.batch_size = batch_size
        self.vocab = vocab
        self.mode = mode

        self.batcher = self.get_batcher()
        self.batch = None

        self.num_batches = int(np.ceil(len(self.dataset) / (1.0 * batch_size)))

        self.track_samples = []
        self.track = track

        print(f'Using neural reward function...')
        self.reward_fn = NeuralRewardFunction(
            model_dir=reward_model_dir,
            device=device, use_log_prob=use_log_prob,
            use_reward_table=use_reward_table
        )

        self.collate_fn = vocab.collate_fn

    def update_dataset(self, raw_data):
        self.dataset.raw_data = raw_data
        self.dataset.process_data()
        self.num_batches = int(np.ceil(len(self.dataset) / (1.0 * self.batch_size)))
        self.batcher = self.get_batcher()
        self.batch = None
        self.track_samples = []

    def get_batcher(self):
        indices = list(range(len(self.dataset)))
        if self.mode == 'train':
            random.shuffle(indices)
        batch_size = self.batch_size

        for st in range(0, len(indices), batch_size):
            en = st + batch_size
            batch_idxs = indices[st:en]
            batch = [self.dataset[idx] for idx in batch_idxs]

            yield batch

    def reset(self):
        """
        Each point in data is a (D, G) pair.
        Returns observation which is agent consumable (similar to vocabulary)
        """
        try:
            self.batch = next(self.batcher)
        except StopIteration:
            self.batcher = self.get_batcher()
            self.batch = next(self.batcher)

        return self.collate_fn(self.batch)

    def step(self, actions):
        if self.mode == 'train':
            return self.step_train(actions)
        else:
            return self.step_infer(actions)

    def step_train(self, actions):
        """
        Actions for the current batch. We are expected to return the following
        (observation, reward, done, info). Note that since we are simulating a
        single action episode, done is always returned as true.
        Beware the batch nature.
        :param actions: torch.tensor([(src, rel, dest)]) where src, rel, dest are ids
        """
        actions = actions.detach().cpu().numpy()
        rewards = self.reward_fn(self.batch, actions, mode='train')
        rewards = torch.tensor(rewards)

        return None, rewards, [True] * len(actions), None

    def step_infer(self, actions, track_all=False):
        """
        Actions for the current batch. We are expected to return the following
        (observation, reward, done, info). Note that since we are simulating a
        single action episode, done is always returned as true.
        Beware the batch nature.
        :param actions: torch.tensor([(src, rel, dest)]) where src, rel, dest are ids
        """
        actions = actions.detach().cpu().numpy()
        rewards = self.reward_fn(self.batch, actions, mode='infer')
        track_idxs = []

        for idx in range(len(actions)):
            obj = self.batch[idx]
            if (int(obj['did']) in dids_to_track) or track_all:
                track_idxs.append(idx)

        for idx in track_idxs:
            self.track_samples.append(deepcopy(self.batch[idx]))
            self.track_samples[-1]['reward'] = rewards[idx]
            self.track_samples[-1]['action'] = actions[idx]

        rewards = torch.tensor(rewards)

        return None, rewards, [True] * len(actions), None

    def get_tracking_samples(self, do_print=False):
        if not self.track:
            self.track_samples = []
            return None

        prints = []
        for obj in self.track_samples:
            pred = obj['action']
            reward = obj['reward']

            ret = {
                'did': obj['did'],
                'source': obj['source'],
                'decision': int(pred),
                'reward': reward,
                'sign': obj['sign']
            }

            prints.append(ret)

        if do_print:
            print(json.dumps(prints, indent=2))

        self.track_samples = []

        return prints


class MAPOEnvironment(Environment):
    def __init__(
        self, fname, vocab, mode='train', batch_size=1, track=False,
        reward_model_dir='', device='cpu',
        use_log_prob=False, buff_clip=0.2, sample_size=100,
        use_reward_table=None
    ):
        super().__init__(
            fname, vocab, mode, batch_size, track,
            reward_model_dir, device, use_log_prob,
            use_reward_table=use_reward_table
        )
        self.model = None
        self.device = device

        self.explored_trajectories = set()  # Reward known for these trajectories
        self.good_traj_data = defaultdict(lambda: [])  # Good trajectories explored so far
        self.good_trajectories = set()  # Good trajectory Ids explored so far
        self.state_wts = defaultdict(lambda: 0.0)
        self.buff_clip = buff_clip
        self.training_batches = []
        self.batcher = None
        self.sample_size = sample_size

    def set_model(self, model):
        if self.model is not None:
            del self.model

        self.model = model
        self.model.set_sampling_mode(True)
        self.model.eval()
        self.model.to(self.device)
        print('Updating Buffers')
        self.update_buffer()
        print('Processing Data')
        self.process_data()

    def update_buffer(self):
        """
        Update weights of explored good trajectories
        """
        data = []
        collate_fn = self.collate_fn

        for sign in self.good_traj_data:
            data.extend(self.good_traj_data[sign])
            self.state_wts[sign] = 0.0

        print(f'Buffer size {len(data)}')
        batch_size = 256
        all_probs = []
        t_iter = list(range(0, len(data), batch_size))
        for st in t_iter:
            en = st + batch_size
            obs = data[st:en]
            tbatch = [
                x.to(self.device) for x in collate_fn(obs)
            ] + [True]

            with torch.no_grad():
                _, log_probs = self.model(*tbatch)
                probs = torch.exp(log_probs)
                probs = probs.detach().cpu().numpy()

            all_probs.extend(probs)

        sidx = 0
        for sign in self.good_traj_data:
            eidx = len(self.good_traj_data[sign])
            self.state_wts[sign] = np.sum(all_probs[sidx:eidx])

            for obj in self.good_traj_data[sign]:
                obj['prob'] = all_probs[sidx]
                obj['wt'] = max(self.buff_clip, self.state_wts[sign])
                sidx += 1

    def process_batch(self, batch):
        all_actions = [0, 1]
        rating_cands = list(range(len(all_actions)))
        collate_fn = self.collate_fn

        # 1. compute good trajectories
        data = []
        for obj in batch:
            sign = obj['sign']
            not_explored_actions = []
            for action in rating_cands:
                if (sign, action) not in self.explored_trajectories:
                    not_explored_actions.append(action)

            if len(not_explored_actions) == 0:
                break

            states = [deepcopy(obj) for _ in range(len(not_explored_actions))]
            rewards = self.reward_fn(states, not_explored_actions, mode='train')

            for idx, action in enumerate(not_explored_actions):
                self.explored_trajectories.add((sign, action))

                if rewards[idx] < 0:
                    continue

                tobj = deepcopy(obj)
                tobj['target'] = action
                tobj['reward'] = rewards[idx]
                data.append(tobj)
                self.good_trajectories.add((sign, action))

        # 2. Compute wts.
        if len(data) > 0:
            obs = collate_fn(data)
            tbatch = [x.to(self.device) for x in obs]
            tbatch = tbatch + [True]

            with torch.no_grad():
                _, log_probs = self.model(*tbatch)
                probs = torch.clamp(torch.exp(log_probs), min=1e-4)
                probs = probs.detach().cpu().numpy()

            for idx, obj in enumerate(data):
                sign = obj['sign']
                obj['prob'] = probs[idx]
                self.state_wts[sign] += probs[idx]
                # obj['wt'] = max(self.buff_clip, self.state_wts[sign])
                self.good_traj_data[sign].append(obj)

        # 3. on policy samples
        obs = collate_fn(batch)
        tbatch = [x.to(self.device) for x in obs]
        tbatch = tbatch + [False]
        with torch.no_grad():
            actions, log_probs = self.model(*tbatch)
            probs = torch.exp(log_probs)
            probs = probs.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()

        trewards = self.reward_fn(batch, actions, mode='train')

        final_data = []
        for idx, obj in enumerate(batch):
            sign = obj['sign']

            # i. Get good sample
            if len(self.good_traj_data[sign]) > 0:
                # good_sample = max(self.good_traj_data[sign], key=lambda x: x['prob'])
                probs = np.array([x['prob'] for x in self.good_traj_data[sign]])

                if np.sum(probs) == 0:
                    probs = np.ones(len(probs)) / len(probs)

                probs = probs / np.sum(probs)
                jdx = np.random.choice(a=len(probs), p=probs)
                good_sample = deepcopy(self.good_traj_data[sign][jdx])
                good_sample['wt'] = max(self.buff_clip, self.state_wts[sign])
                final_data.append(good_sample)

            # ii. Get onpolicy sample
            action = actions[idx]
            if (sign, action) in self.good_trajectories:
                continue

            tobj = deepcopy(obj)
            tobj['target'] = action
            tobj['reward'] = trewards[idx]
            tobj['wt'] = 1.0 - max(self.buff_clip, self.state_wts[sign])
            final_data.append(tobj)

        return final_data

    def process_data(self):
        self.training_batches = []        
        batcher = self.get_batcher()

        for batch in batcher:
            new_batch = []
            for _ in range(self.sample_size):
                tbatch = self.process_batch(batch)
                new_batch.extend(tbatch)
                
            self.training_batches.append(new_batch)

        self.batcher = None
        self.num_batches = len(self.training_batches)

    def get_buffer_dataloader(self):
        random.shuffle(self.training_batches)

        from collections import Counter
        print(Counter([x['target'] for y in self.training_batches for x in y]))

        for batch in self.training_batches:
            yield batch

    def reset(self):
        collate_fn = self.collate_fn

        if self.batcher is None:
            self.batcher = self.get_buffer_dataloader()

        try:
            batch = next(self.batcher)
        except StopIteration:
            self.batcher = self.get_buffer_dataloader()
            batch = next(self.batcher)

        obs = collate_fn(batch)
        rewards = torch.tensor([x['reward'] for x in batch])
        wts = torch.tensor([x['wt'] for x in batch])

        return obs, rewards, wts
