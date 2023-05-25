import os, json
import joblib
from model import EntryRemoverAgent

from utils import load_best_model, read_cli, train_file
from environment.environments import Environment
from trainer import RLTrainer

args = read_cli()
print(args)
os.makedirs(args['dest_loc'], exist_ok=True)


if args['dataset'] == 'babi':
    from vocabulary.babi_agent_vocab import Vocabulary
elif args['dataset'] == 'bitod':
    from vocabulary.bitod_agent_vocab import Vocabulary
else:
    raise NotImplementedError


data_loc = args['data_loc']
vocab = joblib.load(os.path.join(args['dest_loc'], 'vocab.pkl'))

# 2. Get dataloaders
data_loc = args['data_loc']
batch_size = args['batch_size']
train_fname = os.path.join(data_loc, train_file)

# 3. Get Environment
train_fname = os.path.join(args['data_loc'], train_file)
eval_env = Environment(
    train_fname, vocab, mode='infer', batch_size=args['batch_size'],
    reward_model_dir=args['reward_model_loc'],
    device=args['device'], use_reward_table='train', track=True
)
args.update(vocab.get_ov_config(args))
model = load_best_model(args, EntryRemoverAgent)

trainer = RLTrainer(model, None, None, eval_env, device=args['device'])

ret, logs = trainer.eval_results()
print(ret)
predictions = []
rem_cnt = sum([1 for x in logs if x['decision'] == 0])

print(f'Removed {rem_cnt} entries from the KB')

fname = os.path.join(args['data_loc'], train_file[:-5] + '_pred.json')
with open(fname, 'w') as fp:
    json.dump(logs, fp, indent=2)
