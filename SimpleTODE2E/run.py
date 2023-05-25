import os
import json
import pickle

import numpy as np
import torch
from transformers import (
    EarlyStoppingCallback, IntervalStrategy, TrainingArguments
)

from dataset import BasicDataset
from log_utils import setup_logging
from models import GPT2TOD, BloomTOD
from trainers import TODTrainer as Trainer
from vocab import CausalVocab
from utils import get_config, override_config, load_json, read_cli, train_tag, val_tag
import wandb


def run(args, report_to='tensorboard'):
    # 0. Set seeds and Logs
    seed = args['train']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    setup_logging(args)

    model_type = args['model']['type']
    wildcard = args['model']['wildcard']

    if model_type == 'gpt2':
        Model = GPT2TOD
        Vocabulary = CausalVocab
    elif model_type == 'bloom':
        Model = BloomTOD
        Vocabulary = CausalVocab
    else:
        raise NotImplementedError

    # 1. Get vocabulary
    vocab = Vocabulary(args)
    train_file = os.path.join(args['datapath'], train_tag)
    vocab.fit(load_json(train_file))

    dest_path = os.path.join(args['destpath'], args['experiment_name'])
    resume = args['train'].get('resume_training', False)
    if os.environ.get('LOCAL_RANK', '-1') in ['-1', '0']:
        os.makedirs(dest_path, exist_ok=resume)
        with open(os.path.join(dest_path, 'vocab.pkl'), 'wb') as fp:
            pickle.dump(vocab, fp)

        with open(os.path.join(dest_path, 'run_config.json'), 'w') as fp:
            json.dump(args, fp, indent=2)

    # 2. Get model
    ov_cfg = vocab.get_ov_config()
    model = Model.from_pretrained(wildcard, cfg=ov_cfg)
    model.resize_token_embeddings(vocab.total_vocab_size)

    # 3. Get datasets
    train_dataset = BasicDataset(train_file, vocab, mode='train')
    val_file = os.path.join(args['datapath'], val_tag)
    val_dataset = BasicDataset(val_file, vocab, mode='infer')

    # 4. Setup trainer arguments
    train_args = TrainingArguments(
        output_dir=dest_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
        num_train_epochs=args['train']['num_epochs'],
        learning_rate=args['train']['learning_rate'],
        max_steps=args['train'].get('max_steps', -1),
        # save_strategy='epoch',
        save_strategy=IntervalStrategy.STEPS,
        save_steps=args['train'].get('save_eval_steps', 100),
        seed=args['train']['seed'],
        fp16=args['train']['fp16'],
        # evaluation_strategy="epoch",
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=args['train'].get('save_eval_steps', 100),
        gradient_accumulation_steps=args['train']['gradient_accumulation_steps'],
        logging_steps=1,
        ddp_find_unused_parameters=False,
        save_total_limit=args['train'].get('save_total_limit', 5),
        load_best_model_at_end=True,
        metric_for_best_model=args['train']['metric_for_best_model'],
        greater_is_better=args['train']['greater_is_better'],
        report_to=report_to,
        run_name=args['experiment_name'],
        warmup_ratio=args['train'].get('warmup_ratio', 0.0)
    )

    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=vocab.collate_fn,
        cfg=args, vocab=vocab, callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args['train'].get('early_stopping_patience', 1))
        ]
    )
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    cargs = read_cli()
    args = get_config(cargs['config'])
    args = override_config(args, cargs)

    local_rank = os.environ.get('LOCAL_RANK', '')
    report_to = 'tensorboard'
    if args.get('use_wandb', False):
        import wandb
        wandb.init()
        report_to='wandb'
    run(args, report_to=report_to)
