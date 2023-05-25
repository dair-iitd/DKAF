
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from copy import deepcopy

import os
import re
import subprocess
import tempfile
import numpy as np

from collections import defaultdict


# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BLEU metric implementation.
"""


def moses_multi_bleu(hypotheses, references, lowercase=False):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """

    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    
    # Get MOSES multi-bleu script
    multi_bleu_path = os.path.abspath("./multi-bleu.perl")

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score


###################################################################################################################
def compute_babi_metrics(dataset, predictions, ent2type=None):
    all_targets = []
    all_preds = []
    all_dids = []
    for ii, pred in enumerate(predictions):
        all_targets.append(dataset[ii]['output'])
        all_preds.append(pred)
        all_dids.append(dataset[ii]['did'])

    per_did_corr = defaultdict(lambda : 0)
    per_did_num = defaultdict(lambda : 0)

    sample_results = []
    for tar, pred, did in zip(all_targets, all_preds, all_dids):
        per_did_corr[did] += int(tar == pred)
        per_did_num[did] += 1
        sample_results.append({
            'response_accuracy': int(tar == pred)
        })

    corr_dlg, num_dlg = 0, 0
    corr_uttr, num_uttr = 0, 0
    for did in per_did_corr:
        num_dlg += 1
        if per_did_corr[did] == per_did_num[did]:
            corr_dlg += 1

        corr_uttr += per_did_corr[did]
        num_uttr += per_did_num[did]

    dlg_acc = (100.0 * corr_dlg) / num_dlg
    uttr_acc = (100.0 * corr_uttr) / num_uttr

    metrics = {
        'dialog_accuracy': dlg_acc,
        'utterance_accuracy': uttr_acc,
    }

    return metrics, sample_results
    

def get_entities_from_utterance(uttr, gold_entities, ent2type):
    entities = []
    for word in uttr.split():
        if word in ent2type or word in gold_entities:
            entities.append(word)

    return entities


def compute_f1(gold_entities, pred_entities):
    tp, fp, fn = 0, 0, 0
    for ent in pred_entities:
        if ent not in gold_entities:
            fp += 1
    
    for ent in gold_entities:
        if ent in pred_entities:
            tp += 1
        else:
            fn += 1

    prec = 0
    if (tp + fp) > 0:
        prec = tp / (tp + fp)
    rec = 0
    if (tp + fn) > 0:
        rec = tp / (tp + fn)
    f1 = 0
    if (prec + rec) > 0:
        f1 = (2 * prec * rec) / (prec + rec)

    return prec, rec, f1


def compute_metrics_sample(sample, pred, ent2type):
    metrics = {
        'ent_f1': 0, 'ent_f1_valid': 0,
        'kb_f1': 0, 'kb_f1_valid': 0,
        'ctx_f1': 0, 'ctx_f1_valid': 0,
    }

    gold_entities = sample['gold_entities']
    if len(gold_entities) == 0:
        return metrics

    pred_entities = get_entities_from_utterance(pred, gold_entities, ent2type)
    prec, rec, f1 = compute_f1(gold_entities, pred_entities)
    metrics['ent_f1'], metrics['ent_f1_valid'] = f1, 1

    # KB and CTX
    ctx_tokens = [tt for uu in sample['context'] for tt in uu.split()]
    kb_tokens = [val for ee in sample['kb'] for val in ee.values()]

    kb_pred_entities = []
    ctx_pred_entities = []
    for ent in pred_entities:
        if ent in kb_tokens and ent not in ctx_tokens:
            kb_pred_entities.append(ent)
        else:
            ctx_pred_entities.append(ent)

    kb_gold_entities = []
    ctx_gold_entities = []
    for ent in gold_entities:
        if ent in kb_tokens and ent not in ctx_tokens:
            kb_gold_entities.append(ent)
        else:
            ctx_gold_entities.append(ent)

    if len(kb_gold_entities) > 0:
        prec, rec, f1 = compute_f1(kb_gold_entities, kb_pred_entities)
        metrics['kb_f1'], metrics['kb_f1_valid'] = f1, 1

    if len(ctx_gold_entities) > 0:
        prec, rec, f1 = compute_f1(ctx_gold_entities, ctx_pred_entities)
        metrics['ctx_f1'], metrics['ctx_f1_valid'] = f1, 1

    return metrics


def compute_bitod_metrics(dataset, predictions, ent2type):
    sample_results = []
    for ii, pred in enumerate(predictions):
        ret = compute_metrics_sample(dataset[ii], pred, ent2type)
        sample_results.append(ret)

    tmp = {
        'ent_f1': 0, 'ent_f1_valid': 0,
        'kb_f1': 0, 'kb_f1_valid': 0,
        'ctx_f1': 0, 'ctx_f1_valid': 0,
    }
    domain_tmp = {
        'hotels': deepcopy(tmp),
        'restaurant': deepcopy(tmp),
        'attraction': deepcopy(tmp),
    }
    for ii, entry in enumerate(sample_results):
        dom = dataset[ii]['type']
        for key, val in entry.items():
            tmp[key] += val
            domain_tmp[dom][key] += val

    metrics = {
        'entity_f1': round(tmp['ent_f1'] / tmp['ent_f1_valid'], 5),
        'kb_f1': round(tmp['kb_f1'] / tmp['kb_f1_valid'], 5),
        'ctx_f1': round(tmp['ctx_f1'] / tmp['ctx_f1_valid'], 5),
    }

    for dom in ['hotels', 'restaurant', 'attraction']:
        tmp = domain_tmp[dom]
        tmetrics = {
            f'{dom}_entity_f1': round(tmp['ent_f1'] / tmp['ent_f1_valid'], 5),
            f'{dom}_kb_f1': round(tmp['kb_f1'] / tmp['kb_f1_valid'], 5),
            f'{dom}_ctx_f1': round(tmp['ctx_f1'] / tmp['ctx_f1_valid'], 5),
        }
        metrics.update(tmetrics)

    all_hyps = []
    all_refs = []
    for ii, pred in enumerate(predictions):
        target = dataset[ii]['output']
        pred = predictions[ii]
        all_refs.append(target)
        all_hyps.append(pred)

    metrics['bleu'] = round(moses_multi_bleu(all_hyps, all_refs), 5)

    return metrics, sample_results
