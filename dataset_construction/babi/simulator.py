import os
import random
import argparse
import numpy as np
from copy import deepcopy

from clock import CalendarClock
from temporal_engine import bAbIEngine
from babi_Restaurant import load_kb
from utils import read_babi_dialogs, write_dialogs_data, parse_results

from rewriter import DynamicDialogRewriter

kb_loc = "./yelp_dialog_babi_kb.pkl"
task_dialog_files = {
    'dev': 'dialog-babi-task5-full-dialogs-dev.txt',
    'train': 'dialog-babi-task5-full-dialogs-trn.txt',
    'oov-test': 'dialog-babi-task5-full-dialogs-tst-OOV.txt',
    'test': 'dialog-babi-task5-full-dialogs-tst.txt'
}


def read_cli():
    parser = argparse.ArgumentParser(description="bAbI Evolving KB Simulator")
    parser.add_argument(
        "--data_loc",
        help="Original bAbI data location",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--dest_loc",
        help="Destination folder where simulated dialogs will be created",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="Seed value for the simulation",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--use_latest_kb",
        help="Whether to use recent KB while saving the simulated dialogs. Setting this parameter to True will lead to inconsistent dialogs. Otherwise, dialogs will have their contemporary KB.",
        required=True,
        type=str,
        choices=['True', 'False']
    )
    parser.add_argument(
        "--start_date",
        help="Start Date (yyyy-mm-dd) to be used for simulating evolving KB",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--end_date",
        help="End Date (yyyy-mm-dd) to be used for simulating evolving KB",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--clock_resolution",
        help="Clock Resolution (min) to be used for simulating evolving KB",
        required=True,
        type=int,
    )

    args = vars(parser.parse_args())
    args['use_latest_kb'] = args['use_latest_kb'] == 'True'

    return args


class bAbISimulator(object):
    def __init__(self, clock, engine, rewriter):
        self.clock = clock
        self.engine = engine
        self.rewriter = rewriter
        self.all_entities = engine.get_all_entities()

    def rewrite_dialog(self, dialog):
        # 1. Extract API call
        # 2. Query the engine
        # 3. Rewrite the dialog
        # Make sure that we get "consistent copy for now."
        api_call = dialog.api_call
        api_results = self.engine.select(api_call)

        if len(api_results['kb']) > 0:
            new_dialog = self.rewriter.rewrite_dialog(
                dialog, api_results
            )

            return new_dialog

        else:
            return None

    def run(self, data):
        idx = 0
        data_size = len(data)
        new_dialogs = []

        completed_dids = set()
        random.shuffle(data)

        event = self.clock.tick()
        while event is not None:
            while idx in completed_dids and len(completed_dids) < len(data):
                idx = (idx + 1) % data_size
                
            self.engine.step(event)
            dlg = self.rewrite_dialog(data[idx])

            if dlg is not None:
                new_dialogs.append(dlg)
                completed_dids.add(idx)

            event = self.clock.tick()

            if len(completed_dids) == len(data):
                break

        return new_dialogs

    def use_last_kb(self, data):
        curr_time = deepcopy(self.clock.curr_time)
        final_api_results = dict()

        for dialog in data:
            api_call = dialog.api_call
            if api_call in final_api_results:
                continue

            self.clock.curr_time = curr_time
            while True:
                event = self.clock.tick()
                self.engine.step(event)
                api_results = self.engine.select(api_call)

                if len(api_results['kb']) > 0:
                    break

            final_api_results[api_call] = api_results

        for dialog in data:
            api_call = dialog.api_call
            api_results = deepcopy(final_api_results[api_call])
            dialog.api_results = api_results
            dialog.api_results['kb'] = parse_results(dialog.api_results['kb'])

        return data


def run(args):
    seed = args['seed']
    np.random.seed(seed)
    random.seed(seed)

    key = 'train'
    clk_cfg = {
        'step_size': args['clock_resolution'],
        'start_date': args['start_date'],
        'end_date': args['end_date'],
    }
    clock = CalendarClock(clk_cfg)
    zero_ts = clock.get_clock_start()

    data_loc = args['data_loc']
    kb_file = os.path.join(data_loc, kb_loc)
    resto_list = load_kb(kb_file)

    engine = bAbIEngine(resto_list, zero_ts)

    rewriter = DynamicDialogRewriter()
    sim = bAbISimulator(clock, engine, rewriter)

    file = os.path.join(data_loc, task_dialog_files[key])
    data = read_babi_dialogs(file)

    rwr_data = sim.run(data)
    print(f'Number of rewritten dlgs {len(rwr_data)}')

    from babi_Restaurant import closed_cnt
    print(f'Number of restaurant closed {closed_cnt}')

    from babi_Restaurant import access_cnt
    print(f'Number of restaurant accessed {access_cnt}')

    print(f'Number of restaurants {engine.count_accessed_samples()}')

    if args['use_latest_kb']:
        rwr_data = sim.use_last_kb(rwr_data)

    [x.set_entities() for x in rwr_data]

    os.makedirs(args['dest_loc'], exist_ok=True)
    dname = os.path.join(args['dest_loc'], f'{key}.txt')
    write_dialogs_data(rwr_data, dname)


if __name__ == "__main__":
    args = read_cli()
    run(args)
