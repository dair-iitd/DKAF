import copy
import numpy as np
import random
import pickle
import copy


OUTAGE_PROB = 0.05
SUDDEN_DEATH_PROB = 1e-5
SUDDEN_AVAIL_PROB = 0.05
OUTAGE_TIME = 1  # day

MIN_THRESHOLD = 0.7
MAX_THRESHOLD = 0.75

closed_cnt = 0
access_cnt = 0


class bAbIRestaurant(object):
    
    def __init__(self, attributes, schedule, table_size=5):
        self.attributes = attributes

        exp_cols = {
            'R_cuisine', 'R_location',
            'R_number', 'R_price', 'R_phone',
            'R_address', 'R_restro', 'R_rating'
        }
        cols = set(attributes.keys())
        diff = exp_cols - cols
        if len(diff) > 0:
            print(f"Columns {diff} missing in the attributes.")

        # Static Attributes - Name, Rating, phone, address, cuisine, table_size
        # Dynamic Attributes - Schedule, Tables_available, (Outage, Sudden Death) can be handled in step function

        self.timestamp = None
        self.available = True
        self.table_avail = table_size
        self.table_size = table_size
        self.schedule = schedule
        self.threshold = None
        self.missing_cnt = False

        # Permanently closed
        self.closed = False

        # Outage
        self.temp_close = False
        self.outage_end = None

        # accessed
        self.accessed = False

        self.compute_threshold()

    def compute_threshold(self):
        cnts = []
        for ctimes in self.schedule.values():
            for val in ctimes.values():
                cnts.append(val)

        maxcnt = max(cnts)
        assert max(cnts) != 1

        val = np.random.rand()
        val = MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * val
        val = np.floor(val * maxcnt)

        self.threshold = val

    def reset(
        self, timestamp, available=None, table_avail=None,
        table_size=None, schedule=None, closed=None
    ):
        self.timestamp = timestamp

        if available is not None:
            self.available = available

        if table_avail is not None:
            self.table_avail = table_avail

        if table_size is not None:
            self.table_size = table_size

        if schedule is not None:
            self.schedule = schedule

        if closed is not None:
            self.closed = closed

    def enquire(self, query):
        ret = all(
            self.attributes[key] == value
            for key, value in query.items()
        ) and self.available and self.table_avail > 0

        if ret and not self.accessed:
            global access_cnt
            access_cnt += 1
            self.accessed = True

        return [ret, self.missing_cnt]

    def to_dict(self):
        return copy.deepcopy(self.attributes)

    def check_availability(self, timestamp):
        return self.schedule[timestamp['day']][timestamp['time']]

    def book_table(self):
        if self.table_avail > 0:
            self.table_avail -= 1
        else:
            print("Unable to book")

    def step(self, timestamp):
        self.timestamp = timestamp
        self.missing_cnt = False

        if self.closed:
            assert self.available == False
            return

        if (
            self.temp_close and\
            timestamp['day'] <= self.outage_end['day'] and\
            timestamp['time_of_the_day'] < self.outage_end['time_of_the_day']
        ):
            assert self.available == False
            return

        self.temp_close = False
        day = timestamp['weekday']
        ts = timestamp['time_of_the_day']

        day_timings = self.schedule[day]
        cnt = day_timings.get(ts, None)

        if cnt is None:
            self.available = True
            self.missing_cnt = True

        elif cnt < self.threshold:
            self.available = True

        else:
            self.available = np.random.rand() > (1.0 - SUDDEN_AVAIL_PROB)

        # overrides
        if np.random.rand() < SUDDEN_DEATH_PROB:
            self.available = False
            self.closed = True

            global closed_cnt
            closed_cnt += 1
            return

        if np.random.rand() < OUTAGE_PROB:
            self.available = False
            self.temp_close = True
            self.outage_end = copy.deepcopy(self.timestamp)
            self.outage_end['day'] += 1

            return


def load_kb(fname):
    print(f'Loading KB from {fname}')

    with open(fname, 'rb') as fp:
        objs = pickle.load(fp)

    print(f"Read {len(objs)} records.")

    exp_cols = {
        'R_cuisine', 'R_location',
        'R_number', 'R_price', 'R_phone',
        'R_address', 'R_restro', 'R_rating'
    }
    resto_list = []
    for obj in objs:
        attributes = dict()
        for key in exp_cols:
            attributes[key] = obj[key]
        resto_list.append(bAbIRestaurant(attributes, obj['timings']))

    return resto_list
