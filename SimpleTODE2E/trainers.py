import json
import torch
import logging
from tqdm import tqdm
from transformers import Trainer

try:
    import wandb
except:
    wandb = None

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from metrics import compute_babi_metrics, compute_bitod_metrics

logger = logging.getLogger()


class TODTrainer(Trainer):
    def __init__(self, cfg, vocab, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.vocab = vocab

    def get_train_dataloader(self):
        logger.info(f"Resetting training dataset....")
        self.train_dataset.process_data()
        return super().get_train_dataloader()

    def decode_responses(self, inputs, outputs):
        tokenizer = self.vocab.lm_tokenizer
        eos = self.vocab.rsp_end

        input_end = inputs['input_ids'].size(1)
        outputs = outputs[:, input_end:]
        preds = []
        responses = tokenizer.batch_decode(
            outputs, clean_up_tokenization_spaces=False
        )
        for resp in responses:
            pred = resp.split(eos, 1)[0]
            preds.append(pred.strip())

        return preds

    def run_evaluation(self, dataset):
        local_rank = self.args.local_rank
        world_size = max(self.args.world_size, 1)

        max_new_tokens = self.cfg['dev']['max_resp_length']
        pred_end = self.vocab.eos_token_idx
        model = self._wrap_model(self.model, training=False)
        model.eval()

        if type(model) == DataParallel:
            model = model.module

        if world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            sampler=sampler,
        )

        if world_size > 1:
            sampler.set_epoch(0)
        loader = tqdm(dataloader, desc='Evaluation...') if local_rank in [-1, 0] else dataloader
        responses = []
        for inputs in loader:
            batch = dict([(k, v.to(self.model.device)) for k, v in inputs.items()])

            if self.cfg['dev'].get('sample', False):
                outputs = model.generate(
                    **batch, use_cache=True, eos_token_id=pred_end,
                    pad_token_id=pred_end, max_new_tokens=max_new_tokens,
                    do_sample=True, min_length=1,
                    temperature=self.cfg['dev'].get('temperature', 0.85),
                    top_k=self.cfg['dev'].get('top_k', 8),
                    top_p=self.cfg['dev'].get('top_p', 0.9),
                )
            else:
                outputs = model.generate(
                    **batch, use_cache=True, eos_token_id=pred_end,
                    pad_token_id=pred_end, max_new_tokens=max_new_tokens, do_sample=False,
                    num_beams=self.cfg['dev'].get('num_beams', 1)
                )

            outputs = outputs.to('cpu')
            responses.extend(self.decode_responses(inputs, outputs))

        model.train()
        if world_size > 1:
            all_responses = [None for _ in range(self.args.world_size)]
            dist.all_gather_object(all_responses, responses)
        else:
            all_responses = [responses]

        final_responses = []
        for ii in range(len(responses)):
            for resps in all_responses:
                final_responses.append(resps[ii])
        final_responses = final_responses[:len(dataset)]

        return final_responses

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
        save_results=False,
    ):
        logger.info(f'Running evaluation......{self.args.local_rank}')
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        responses = self.run_evaluation(eval_dataset)

        if self.cfg['dataset'] == 'babi':
            tmetrics, sample_results = compute_babi_metrics(eval_dataset, responses)
        elif self.cfg['dataset'] == 'bitod':
            tmetrics, sample_results = compute_bitod_metrics(eval_dataset, responses, self.vocab.ent2type)
        else:
            raise NotImplementedError

        metrics = dict()
        for key in tmetrics:
            metrics[f"{metric_key_prefix}_{key}"] = tmetrics[key]
        if wandb is not None and wandb.run is not None:
            wandb.log(metrics)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        logger.info(f'Evaluation results: {json.dumps(metrics, indent=2)}')

        if save_results and self.args.local_rank in [-1, 0]:
            ret = []
            for ii, resp in enumerate(responses):
                ret.append(dict())
                ret[-1]['context'] = eval_dataset[ii]['context']
                ret[-1]['kb'] = eval_dataset[ii]['kb']
                ret[-1]['target'] = eval_dataset[ii]['output']
                ret[-1]['prediction'] = resp
                ret[-1].update(sample_results[ii])
            with open('results.json', 'w') as fp:
                json.dump(ret, fp, indent=2)

        return metrics
