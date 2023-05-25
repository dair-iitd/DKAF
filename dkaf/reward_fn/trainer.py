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
from collections import defaultdict


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


def compute_metrics(targets, logits, data=None):
    preds = []

    for idx in range(len(targets)):
        preds.append(np.argmax(logits[idx]))

    corr_cnt = 0
    for idx in range(len(preds)):
        pred = preds[idx]
        target = targets[idx]

        if pred == target:
            corr_cnt += 1

    ret = dict()
    ret['accuracy'] = round(corr_cnt / len(preds), 5)

    for k, v in ret.items():
        ret[k] = round(v, 5)

    if data is None:
        return ret

    per_did_res = defaultdict(lambda: [])
    ent_type_res = defaultdict(lambda: [])
    for idx in range(len(targets)):
        did = data[idx]['did']
        pred = preds[idx]
        target = targets[idx]
        per_did_res[did].append(int(pred == target))

        ent_type = data[idx]['target_entity_type']
        ent_type_res[ent_type].append(int(pred == target))

    corr_dlg = 0
    for did, res in per_did_res.items():
        if len(res) == sum(res):
            corr_dlg += 1
    acc = (1.0 * corr_dlg) / len(per_did_res)
    ret['dialog_accuracy'] = round(acc, 5)

    per_ent_type_accuracy = []
    for ent_type, res in ent_type_res.items():
        tret = dict()
        tcant = len(res)
        tcorr = sum(res)
        acc = (1.0 * tcorr) / tcant
        tret['entity_type'] = ent_type
        tret['total'] = tcant
        tret['corr'] = tcorr
        tret['acc'] = round(acc, 5)
        per_ent_type_accuracy.append(tret)

    ret['per_entity_type_results'] = per_ent_type_accuracy

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
        loss = res[0]
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

    def evaluate(self, ret_preds=False):
        model = self.model
        device = self.device
        model.eval()

        losses = []
        preds, targets = [], []

        val_dataloader = self.val_dataloader
        for batch in val_dataloader:
            batch = [x.to(device) for x in batch]

            with torch.no_grad():
                res = model(*batch)
                logits = res[1].detach().cpu().numpy()
                tars = batch[-1].detach().cpu().numpy()
                lvals = res[0].detach().cpu().numpy()

            losses.append(lvals)
            preds.extend(logits)
            targets.extend(tars)

        losses = np.mean(losses, axis=0)     

        ret = {'epoch': self.epoch}
        ret.update(compute_metrics(targets, preds, val_dataloader.dataset.data))
        print(json.dumps(ret, indent=2))
        print('Validation_Loss =', round(np.mean(losses), 5))
        self.model.train()

        if ret_preds:
            return ret, preds
        else:
            return ret

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

    def evaluate_model(self, dataloader):
        self.val_dataloader = dataloader
        return self.evaluate(ret_preds=True)
