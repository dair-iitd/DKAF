import torch
import torch.nn as nn

import os
import json
import numpy as np
from tqdm import trange

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
)
from collections import defaultdict, Counter


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


def compute_dialog_accuracy(preds_arr, targets_arr, data):
    did_stats = defaultdict(lambda: [])
    for idx, obj in enumerate(data):
        pred = preds_arr[idx]
        target = targets_arr[idx]
        did = obj['did']

        corr = int(np.array_equal(pred, target))
        did_stats[did].append(corr)

    cnt = 0
    for res in did_stats.values():
        total = len(res)
        corr = sum(res)

        cnt += 1 if total == corr else 0

    print(f'Number of correct dialog {cnt} / {len(did_stats)}')

    return (1.0 * cnt) / len(did_stats)


def compute_metrics(logits_arr, targets_arr, data=None):
    assert len(logits_arr) == len(targets_arr)

    logits_arr = np.array(logits_arr)
    targets_arr = np.array(targets_arr)
    corr_cnt = 0.0
    for idx in range(len(logits_arr)):
        logits = logits_arr[idx]
        targets = targets_arr[idx]

        local_preds, local_tars = [], []
        for jdx in range(len(logits)):
            if logits[jdx] > 0:
                local_preds.append(jdx)
            if targets[jdx] > 0:
                local_tars.append(jdx)
        
        if set(local_tars) == set(local_preds):
            corr_cnt += 1

    overall_acc = corr_cnt / len(logits_arr)

    ret = dict()
    ret['overall_accuracy'] = float(round(overall_acc, 5))

    preds_arr = (logits_arr > 0).astype(np.int64)
    for lab in range(logits_arr.shape[1]):
        preds = preds_arr[:, lab]
        tars = targets_arr[:, lab]

        acc = accuracy_score(tars, preds)
        prec = precision_score(tars, preds)
        rec = recall_score(tars, preds)
        f1 = f1_score(tars, preds)

        ret[lab] = {
            'accuracy': float(round(acc, 5)),
            'relation': float(round(lab, 5)),
            'precision': float(round(prec, 5)),
            'recall': float(round(rec, 5)),
            'f1': float(round(f1, 5)),
            'total': int(np.sum(tars)),
            'total_preds': int(np.sum(preds)),
            # 'counts': list(Counter(tars).most_common())
        }

    ret['accuracy'] = float(round(compute_dialog_accuracy(
        preds_arr, targets_arr, data
    ), 5))

    return ret


class Trainer(object):
    def __init__(
        self, model, optimizer=None, num_epochs=None, vocab=None,
        train_dataloader=None, val_dataloader=None, scheduler=None,
        outdir='./data/', epoch_callback=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.vocab = vocab

        self.device, self.num_gpus = get_device('cuda')
        self.model.to(self.device)

        self.gstep = 0
        self.epoch = 0

        self.save_top = None

        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

        self.epoch_callback = epoch_callback

    def single_step(self, model, optimizer, scheduler, device, batch):
        optimizer.zero_grad()

        batch = [x.to(device) for x in batch]
        res = model(*batch)
        loss = res[-1]
        loss.backward()

        optimizer.step()
        if scheduler:
            scheduler.step(self.gstep)

        self.gstep += 1

        return loss.item()

    def train(self):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        device = self.device

        train_dataloader = self.train_dataloader

        updates_per_epoch = len(train_dataloader)
        total_updates = updates_per_epoch * self.num_epochs + 1
        t_iter = trange(total_updates, desc='Training_Loss: ', leave=True)
        train_data_iter = iter(train_dataloader)
        model.train()

        for idx in t_iter:
            try:
                batch = next(train_data_iter)
            except StopIteration:
                print(f'Epoch {self.epoch} completed.')
                ret = self.evaluate(ret_preds=False)

                ret['epoch'] = self.epoch
                dump_to_db(ret, self.outdir)
                self.create_checkpoint()
                self.epoch += 1
                # train_dataloader.dataset.process_data()
                train_data_iter = iter(train_dataloader)
                batch = next(train_data_iter)

                if self.epoch_callback is not None:
                    self.epoch_callback(
                        self.epoch, model, optimizer, scheduler
                    )

            loss = self.single_step(model, optimizer, scheduler, device, batch)
            desc = f'Training_Loss: {loss}'
            t_iter.set_description(desc=desc, refresh=True)

    def evaluate(self, ret_preds=False, use_metrics=True):
        model = self.model
        device = self.device
        model.eval()

        logits_arr = []
        targets_arr = []
        losses = []

        val_dataloader = self.val_dataloader
        for batch in val_dataloader:
            batch = [x.to(device) for x in batch]

            with torch.no_grad():
                res = model(*batch)
                logits = res[0].detach().cpu().numpy()
                tloss = res[1].detach().item()
                targets = batch[-1].detach().cpu().numpy()

            logits_arr.extend(logits)
            targets_arr.extend(targets)
            losses.append(tloss)

        ret = None
        if use_metrics:
            ret = compute_metrics(
                logits_arr, targets_arr, val_dataloader.dataset.raw_data
            )
            print(json.dumps(ret, indent=2))
            print('Validation_Loss =', round(np.mean(losses), 5))

        self.model.train()

        if not ret_preds:
            return ret
        else:
            return ret, logits_arr, targets_arr

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

    def evaluate_model(self, dataloader, use_metrics=True):
        self.val_dataloader = dataloader
        return self.evaluate(ret_preds=True, use_metrics=use_metrics)
