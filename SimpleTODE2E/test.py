import os
import json
import pickle

from dataset import BasicDataset
from utils import get_config, read_cli, train_tag, val_tag, test_tag

from models import GPT2TOD, BloomTOD
from trainers import TODTrainer as Trainer
from transformers import TrainingArguments

os.environ["WANDB_DISABLED"] = "true"


def run(args):
    print(f"Loading vocab from {args['destpath']}/vocab.pkl")
    with open(os.path.join(args['destpath'], 'vocab.pkl'), 'rb') as fp:
        vocab = pickle.load(fp)

    model_type = args['model']['type']
    if model_type == 'gpt2':
        Model = GPT2TOD
    elif model_type == 'bloom':
        Model = BloomTOD
    else:
        raise NotImplementedError

    chkpt_path = os.path.join(args['destpath'], f"checkpoint-{args['chkpt']}")
    print(f"Loading model from {chkpt_path}")
    model = Model.from_pretrained(chkpt_path)

    train_file = os.path.join(args['datapath'], train_tag)
    train_dataset = BasicDataset(train_file, vocab, mode='train')
    val_file = os.path.join(args['datapath'], test_tag)
    # val_file = os.path.join(args['datapath'], val_tag)
    val_dataset = BasicDataset(val_file, vocab, mode='infer')

    train_args = TrainingArguments(
        output_dir=args['destpath'],
        overwrite_output_dir=True,
        per_device_train_batch_size=args['train']['per_device_train_batch_size'],
        # per_device_eval_batch_size=args['dev']['per_device_eval_batch_size'],
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model, args=train_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=vocab.collate_fn, cfg=args, vocab=vocab
    )
    
    # metrics = trainer.evaluate(train_dataset)
    metrics = trainer.evaluate(val_dataset, save_results=True)
    print(metrics)
    metrics['experiment_name'] = args['experiment_name']

if __name__ == "__main__":
    cargs = read_cli()

    model_path = cargs['model_path']
    cfg_path = os.path.join(model_path, f'run_config.json')
    args = get_config(cfg_path)
    args['destpath'] = model_path
    if cargs['datapath'] is not None:
        args['datapath'] = cargs['datapath']
    else:
        del cargs['datapath']

    if cargs['batch_size'] is not None:
        args['dev']['per_device_eval_batch_size'] = cargs['batch_size']
    else:
        del cargs['batch_size']

    args.update(cargs)
    run(args)
