import json
import pandas as pd


class bAbIEngine(object):
    def __init__(self, resto_list, timestamp):
        """
        Initialize backed KB engine for temporal dialog rewrite
        :param resto_list: [obj] list of bAbI restaurant objects
        """
        self.resto_list = resto_list
        self.timestamp = None

        self.reset(timestamp)

    @staticmethod
    def parse_query(query):
        """
        Extract quriable attributes from query
        :param query: str
        :return: dict of keys
        """
        tokens = query.split()
        if tokens[0] != 'api_call':
            raise Exception('Invalid Query')

        ret = {
            'R_cuisine': tokens[1],
            'R_location': tokens[2],
            'R_number': tokens[3],
            'R_price': tokens[4],
        }

        return ret

    def retrieve(self, constrains):
        """
        For given query string select appropriate entries in KB
        :param constrains: dict of contrains
        :return: [bAbIRestaurant] list of restaurants
        """
        results = [x.enquire(constrains) for x in self.resto_list]

        if any(x[0] == True and x[1] is None for x in results):
            return []
        
        matches = []
        for idx, res in enumerate(results):
            if res[0] == False:
                continue
            matches.append(self.resto_list[idx])

        # matches = filter(lambda x: x.enquire(constrains), self.resto_list)

        return list([x.to_dict() for x in matches])

    def select(self, query):
        """
        Make select query to the KB.
        :param query: str
        :return: [tuple] list of result tuples
        """
        constrains = self.parse_query(query)
        records = self.retrieve(constrains)

        return {
            'kb': records,
            'timestamp': self.timestamp
        }

    def step(self, timestamp):
        for idx in range(len(self.resto_list)):
            self.resto_list[idx].step(timestamp)
        self.timestamp = timestamp

    def reset(self, timestamp):
        print('Resetting the KB')
        for resto in self.resto_list:
            resto.reset(timestamp)
        self.timestamp = timestamp

    def get_all_entities(self):
        all_entities_list = []

        for entry in self.resto_list:
            all_entities_list.extend(entry.attributes.values())

        all_entities_list = list(set(all_entities_list))

        return all_entities_list

    def write_entites_json(self, fname):
        kb_df = pd.DataFrame([
            x.to_dict() for x in self.resto_list
        ])

        all_entities = {}
        columns = [
            'R_cuisine', 'R_location',
            'R_number', 'R_price', 'R_phone',
            'R_address', 'R_rating'
        ]

        all_entities['name'] = kb_df.R_restro.tolist()
        all_entities_list = kb_df.R_restro.tolist()

        for col in columns:
            ents = kb_df[col].to_list()
            ents = list(set(ents))
            all_entities_list.extend(ents)
            all_entities[col] = ents

        ret = {
            'all_entities': all_entities,
            'all_entities_list': list(set(all_entities_list)),
        }

        with open(fname, 'w') as fp:
            json.dump(ret, fp)

    def count_accessed_samples(self):
        count = 0

        for resto in self.resto_list:
            if resto.accessed:
                count += 1

        return count
