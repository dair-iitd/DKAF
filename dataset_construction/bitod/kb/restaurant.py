import copy
import os
import numpy as np
import pickle
from .base import BaseEngine, api_call_to_sign

OUTAGE_PROB = 0.05
SUDDEN_DEATH_PROB = 1e-5
SUDDEN_AVAIL_PROB = 0.05
OUTAGE_TIME = 1  # day

MIN_THRESHOLD = 0.6
MAX_THRESHOLD = 0.7

closed_cnt = 0
access_cnt = 0


class Restaurant(object):
    def __init__(self, attributes, schedule):
        self.attributes = attributes
        self.timestamp = None
        self.available = True
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
        self, timestamp, available=None,
        table_size=None, schedule=None, closed=None
    ):
        self.timestamp = timestamp

        if available is not None:
            self.available = available

        if table_size is not None:
            self.table_size = table_size

        if schedule is not None:
            self.schedule = schedule

        if closed is not None:
            self.closed = closed

    def check_availability(self):
        ret = self.available

        if ret and not self.accessed:
            global access_cnt
            access_cnt += 1
            self.accessed = True

        return [ret, self.missing_cnt]

    def to_dict(self):
        return copy.deepcopy(self.attributes)

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


class RestaurantEngine(BaseEngine):
    def __init__(self, kb_loc, timestamp, kbsize=8, rating_drop=False):
        kb_file = os.path.join(kb_loc, 'restaurants_db.json')
        supported_api_call = 'restaurants_en_US_search'
        super().__init__(kb_file, supported_api_call)

        data = self.kb_df.copy().to_dict('records')
        self.kb_df['IDX'] = np.arange(self.kb_df.shape[0])
        with open(os.path.join(kb_loc, 'yelp_timings.pkl'), 'rb') as fp:
            timings = pickle.load(fp)

        assert len(timings) >= len(data)
        np.random.shuffle(timings)
        self.resto_list = []
        for obj, tim in zip(data, timings):
            self.resto_list.append(
                Restaurant(obj, tim)
            )

        self.timestamp = None
        self.reset(timestamp)
        self.final_kb_size = kbsize
        self.rating_drop = rating_drop
        self.is_frozen = False
        self.api_call_to_buckets = dict()
        self.max_results_per_bucket = 2
        self.max_bucks_to_drop = 2

    def bucket_sample(self, api_call, results):
        buckets = dict()
        for entry in results:
            rr = int(entry['rating'])
            if rr not in buckets:
                buckets[rr] = []
            buckets[rr].append(entry)

        sign = api_call_to_sign(api_call)
        bucket_ids = self.api_call_to_buckets.get(sign, None)
        if bucket_ids is None:
            ratings = sorted(buckets.keys(), key=lambda x: -x)
            bucket_ids = self.sample_buckets_to_drop(ratings)
            self.api_call_to_buckets[sign] = bucket_ids

        tresults = []
        for bid in bucket_ids:
            ret = buckets.get(bid, [])[:self.max_results_per_bucket]
            if len(ret) + len(tresults) > self.final_kb_size:
                break
            tresults.extend(ret)

        return tresults

    def sample_buckets_to_drop(self, buckets):
        if len(buckets) == 1:
            return buckets

        max_bucks_to_drop = self.max_bucks_to_drop
        if len(buckets) == 2:
            max_bucks_to_drop = 1

        buckets = sorted(buckets, key=lambda x: -x)
        for ii in range(max_bucks_to_drop):
            if np.random.rand() <= self.rating_drop:
                buckets[ii] = None
        buckets = [x for x in buckets if x is not None]

        return buckets

    def query(self, api_call):
        results = super().query(api_call)
        results = self.bucket_sample(api_call, results)
        tresults = []
        for obj in results:
            idx = obj['IDX']
            res = self.resto_list[idx].check_availability()
            if res[0] == True and res[1] is None:
                return []
            if res[0] == True:
                tresults.append(obj)
        results = tresults
        results = self.postprocess_results(
            results, api_call, result_size=self.final_kb_size
        )

        return results

    def step(self, timestamp):
        for idx in range(len(self.resto_list)):
            self.resto_list[idx].step(timestamp)
        self.timestamp = timestamp

    def reset(self, timestamp):
        print('Resetting the KB')
        for resto in self.resto_list:
            resto.reset(timestamp)
        self.timestamp = timestamp

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False
