import os
import json
import argparse
import random
import numpy as np
from utils import read_dialogs

from clock import CalendarClock
from kb.restaurant import RestaurantEngine


def read_cli():
    parser = argparse.ArgumentParser(description="Hotel perturbation")
    parser.add_argument(
        "-src_file",
        "--src_file",
        help="Hotel target file",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-kb_loc",
        "--kb_loc",
        help="KB location",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-tar_loc",
        "--tar_loc",
        help="Hotel target location",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-ksz", "--kbsize", help="KB Size",
        required=False,
        type=int,
        default=8,
    )
    parser.add_argument(
        "-rating_drop",
        "--rating_drop",
        help="Rating Drop",
        required=False,
        type=float,
        default=0.1
    )
    parser.add_argument(
        "-start_date",
        "--start_date",
        help="Start Date (yyyy-mm-dd)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-end_date",
        "--end_date",
        help="End Date (yyyy-mm-dd)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "-clock_resolution",
        "--clock_resolution",
        help="Clock Resolution (min)",
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
    args = vars(parser.parse_args())
    args['use_latest_kb'] = args['use_latest_kb'] == 'True'

    return args


class RestaurantSimulator(object):
    def __init__(self, clock, engine):
        self.clock = clock
        self.engine = engine

    def rewrite_dialog(self, dialog):
        new_kb = self.engine.query(dialog.api_call)

        if len(new_kb) > 0:
            if dialog.rewrite(new_kb):
                return True
        return False

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
            ret = self.rewrite_dialog(data[idx])

            if ret:
                new_dialogs.append(data[idx])
                completed_dids.add(idx)

            event = self.clock.tick()

            if len(completed_dids) == len(data):
                break

        return new_dialogs

    def use_last_kb(self, data):
        for idx in range(len(data)):
            data[idx].kb = self.engine.query(data[idx].api_call)

        return data


def run_restaurants(args):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    src_file = args['src_file']
    kb_loc = args['kb_loc']
    tar_loc = args['tar_loc']
    kbsize = args['kbsize']
    rating_drop = args['rating_drop']
    tag = os.path.split(src_file)[-1][:-5]

    key = 'train'
    clk_cfg = {
        'step_size': args['clock_resolution'],
        'start_date': args['start_date'],
        'end_date': args['end_date'],
    }
    clock = CalendarClock(clk_cfg)
    zero_ts = clock.get_clock_start()

    data = read_dialogs(src_file)
    data = [obj for obj in data if obj.task == 'restaurant']
    engine = RestaurantEngine(
        kb_loc=kb_loc, timestamp=zero_ts,
        kbsize=kbsize, rating_drop=rating_drop
    )
    sim = RestaurantSimulator(clock, engine)

    data = sim.run(data)
    print(f'Number of rewritten dlgs {len(data)}')

    if args['use_latest_kb']:
        for idx in range(len(data)):
            data[idx].kb = engine.query(data[idx].api_call)

    os.makedirs(tar_loc, exist_ok=True)
    fname = os.path.join(tar_loc, f'{tag}_restaurant.json')
    with open(fname, 'w') as fp:
        json.dump([dlg.to_dict() for  dlg in data], fp, indent=2)


if __name__ == '__main__':
    args = read_cli()

    run_restaurants(args)
