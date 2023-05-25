from copy import deepcopy
from collections import Counter
import torch

import json
import os
import numpy as np
from tqdm import trange, tqdm


def dump_to_db(obj, dest_loc):
    from tinydb import TinyDB

    fname = os.path.join(dest_loc, 'logs.db')
    db = TinyDB(fname)
    db.insert(obj)

    db.close()


def get_device(device_type='cuda'):
    assert device_type in ['cpu', 'cuda']

    device = torch.device('cpu')
    num_gpus = 0
    if device_type == 'cpu':
        return device, 0

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print('No GPU devices found. Falling back to CPU.')

        return device, 0

    print(f'Found {num_gpus} GPU devices.')
    device = torch.device('cuda')

    return device, num_gpus


def compute_metrics(targets, preds):
    corr_cnt = 0
    for idx in range(len(preds)):
        pred = preds[idx]
        target = targets[idx]

        if pred == target:
            corr_cnt += 1

    ret = dict()
    ret['accuracy'] = round(corr_cnt / len(preds), 5)
    print(f"Prediction Counter {Counter(preds).most_common()}")
    print(f"Gold Counter {Counter(targets).most_common()}")

    for k, v in ret.items():
        ret[k] = round(v, 5)

    return ret


class RLTrainer(object):
    def __init__(
        self, model, optimizer, env, eval_env,
        num_epochs=1, device='cpu', sample_size=1,
        off_sample_prob=0.0, outdir=''
    ):
        self.model = model
        self.env = env
        self.eval_env = eval_env
        self.num_epochs = num_epochs
        self.device = device
        self.optimizer = optimizer
        self.sample_size = sample_size
        self.off_sample_prob = off_sample_prob

        self.device, self.num_gpus = get_device(device)
        self.num_gpus = 1
        self.model.to(self.device)

        self.gstep = 0
        self.epoch = 0
        self.save_top = None
        self.scheduler = None
        self.outdir = outdir

    def compute_loss(self, log_probs, rewards):
        loss = -1.0 * torch.mean(log_probs * rewards)

        return loss

    def create_checkpoint(self):
        tag = f'model_{self.epoch}.bin'
        model = self.model
        optimizer = self.optimizer
        artifacts = [model, optimizer]
        if self.scheduler is not None:
            artifacts.append(self.scheduler)

        artifacts = [self.gstep] + [x.state_dict() for x in artifacts]

        path = os.path.join(self.outdir, tag)
        torch.save(artifacts, path)

    def train(self):
        model = self.model
        env = self.env
        device = self.device
        opt = self.optimizer
        num_batches_per_epoch = env.num_batches

        total_updates = num_batches_per_epoch * self.num_epochs
        t_iter = trange(total_updates, desc='Training_Loss: ', leave=True)
        model.train()

        for self.gstep in t_iter:
            if self.gstep % num_batches_per_epoch == 0:
                print(f'Epoch {self.epoch} completed.')
                
                ret = {'epoch': self.epoch}
                ret.update(self.eval())
                print(json.dumps(ret, indent=2))
                dump_to_db(ret, self.outdir)
                self.create_checkpoint()
                self.epoch += 1
                model.train()

            obs = env.reset()
            batch = [x.to(device) for x in obs]
            loss = 0.0
            rewards = 0.0
            for _ in range(self.sample_size):
                force = np.random.sample() < self.off_sample_prob
                tbatch = batch + [force]

                actions, log_probs = model(*tbatch)
                tactions = actions.to('cpu')
                _, batch_rewards, _, _ = env.step(tactions)
                rewards += torch.sum(batch_rewards).item()
                batch_rewards = batch_rewards.to(device)
                loss += self.compute_loss(log_probs, batch_rewards)

            loss = loss / self.sample_size
            loss.backward()
            opt.step()

    def eval(self, ret_results=False):
        model = self.model
        env = self.eval_env
        device = self.device
        model.eval()

        rewards = []
        all_actions = []
        targets = []
        for _ in range(env.num_batches):
            obs = env.reset()
            batch = [x.to(device) for x in obs]
            tbatch = batch + [False]

            with torch.no_grad():
                actions, log_probs = model(*tbatch)
                tars = batch[-1].detach().cpu().numpy()

            targets.extend(tars)
            tactions = actions.to('cpu')
            _, batch_rewards, _, _ = env.step_infer(tactions, track_all=ret_results)
            rewards.extend(batch_rewards.numpy())
            all_actions.extend(actions.cpu().numpy())

        all_actions = np.array(all_actions)
        avg_reward = float(np.mean(rewards))

        ret = compute_metrics(targets, all_actions)
        ret['average_reward'] = round(avg_reward, 5)

        if ret_results:
            logs = env.get_tracking_samples(do_print=False)
            return ret, logs

        env.get_tracking_samples(do_print=True)

        return ret

    def eval_results(self):
        return self.eval(ret_results=True)


class MAPOTrainer(RLTrainer):
    def __init__(
        self, model, optimizer, env, eval_env, num_epochs=1,
        device='cpu', sample_size=1, off_sample_prob=0.5, outdir='',
        mapo_update_freq=100
    ):
        super().__init__(
            model, optimizer, env, eval_env, num_epochs=num_epochs,
            device=device, sample_size=sample_size,
            off_sample_prob=off_sample_prob, outdir=outdir
        )

        self.mapo_update_freq = mapo_update_freq

    def compute_loss(self, log_probs, rewards, wts):
        loss = -1.0 * torch.mean(log_probs * rewards * wts)

        return loss

    def train(self):
        model = self.model
        env = self.env
        device = self.device
        opt = self.optimizer
        mapo_update_freq = self.mapo_update_freq
        self.gstep = 0

        for epoch in range(self.num_epochs):
            if epoch % mapo_update_freq == 0:
                print("Updating model for MAPO")
                model.to("cpu")
                model_copy = deepcopy(model)
                model.to(device)
                env.set_model(model_copy)

            print(f'Epoch {self.epoch} completed.')                
            ret = {'epoch': self.epoch}
            ret.update(self.eval())
            print(json.dumps(ret, indent=2))
            dump_to_db(ret, self.outdir)
            self.create_checkpoint()
            self.epoch += 1
            model.train()

            t_iter = tqdm(list(range(env.num_batches)))
            for _ in t_iter:
                opt.zero_grad()
                obs, rewards, wts = env.reset()
                batch = [x.to(device) for x in obs] + [True]
                _, log_probs = model(*batch)

                rewards = rewards.to(device)
                wts = wts.to(device)
                loss = self.compute_loss(log_probs, rewards, wts)
                loss.backward()
                opt.step()