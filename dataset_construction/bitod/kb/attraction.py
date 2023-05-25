import os
import numpy as np
from .base import BaseEngine, api_call_to_sign

DEFAULT_AVAIL_PROB = 1.0


class Attraction(object):
    def __init__(self, attributes, avail_prob=DEFAULT_AVAIL_PROB):
        self.attributes = attributes
        self.avail_prob = avail_prob
        self.is_frozen = False
        self.is_available = True

    def check_availibility(self):
        if self.is_frozen:
            return self.is_available
        else:
            self.is_available = np.random.rand() <= self.avail_prob
            return self.is_available

    def freeze(self):
        self.is_frozen = True

    def unfreeze(self):
        self.is_frozen = False


class AttractionEngine(BaseEngine):
    def __init__(
        self, kb_loc, 
        avail_prob=DEFAULT_AVAIL_PROB,
        kbsize=8,
        rating_drop=False
    ):
        kb_file = os.path.join(kb_loc, 'attractions_db.json')
        supported_api_call = 'attractions_en_US_search'
        super().__init__(kb_file, supported_api_call)

        data = self.kb_df.copy().to_dict('records')
        self.entry_objs = [Attraction(obj, avail_prob=avail_prob) for obj in data]
        self.kb_df['IDX'] = np.arange(self.kb_df.shape[0])
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
        results = [
            obj for obj in results
            if self.entry_objs[obj['IDX']].check_availibility()
        ]

        results = self.postprocess_results(
            results, api_call, result_size=self.final_kb_size
        )

        return results

    def freeze(self):
        self.is_frozen = True
        for ii in range(len(self.entry_objs)):
            self.entry_objs[ii].freeze()

    def unfreeze(self):
        self.is_frozen = False
        for ii in range(len(self.entry_objs)):
            self.entry_objs[ii].unfreeze()
