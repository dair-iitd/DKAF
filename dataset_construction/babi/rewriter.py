import re
import random
import numpy as np
from copy import deepcopy

suggestion_regex = r'^what do you think of this option:'
next_option_regex = r'sure let me find an other option for you'

rewritable_patterns = {
    # Order is very important here. R_restro matches the previous ones
    'R_phone': r'resto_.*_phone',
    'R_address': r'resto_.*_address',
    'R_restro': r'resto_.*stars',
}

bot_no_results_replies = [
    "sorry i could not find any restaurants with bookings available",
    "currently no restaurants are free for reservation",
    "you might have to try later. everything is booked."
]
final_bot_reply = "you're welcome"

user_no_results_replies = [
    "that's very unfortunate. thanks.",
    "oh man. thanks anyway",
    "i really wanted to go. thanks for trying."
]


def generate_empty_results_utterances():
    ret = []
    uttr = np.random.choice(bot_no_results_replies)
    ret.append(('bot', uttr))

    uttr = np.random.choice(user_no_results_replies)
    ret.append(('user', uttr))
    ret.append(('bot', final_bot_reply))

    return ret


class BaseDialogRewriter(object):
    def __init__(self):
        """
        Basic dialog rewriter for bAbI task
        """
        pass

    @staticmethod
    def detect_rewrite_start(utterances):
        base_idx = 0
        for idx, entry in enumerate(utterances):
            user, uttr = entry
            if re.search(suggestion_regex, uttr) is not None and user == 'bot':
                base_idx = idx
                break

        return base_idx

    @staticmethod
    def update_bot_utterance(uttr, kb_result):
        for key, regex in rewritable_patterns.items():
            if re.search(regex, uttr) is not None:
                return re.sub(regex, kb_result[key], uttr)

        return uttr

    def rewrite(self, utterances, kb_results):
        new_utterances = deepcopy(utterances)
        base_idx = self.detect_rewrite_start(utterances)
        kb_ptr = 0

        for idx in range(base_idx, len(utterances)):
            user, uttr = utterances[idx]

            if user != 'bot':
                continue

            if re.search(next_option_regex, uttr) is not None:
                kb_ptr += 1
                continue

            new_uttr = self.update_bot_utterance(uttr, kb_results[kb_ptr])
            assert new_uttr == uttr
            new_utterances[idx] = (user, new_uttr)

        return new_utterances

    def rewrite_dialog(self, dialog, api_results):
        """
        Rewrite current dialog based on results from KB
        :param dialog: Dialog bAbI dialog instance
        :param api_results: bAbI API results for rewriting
        :return: dict similar to input with rewritten dialogs
        """
        # 1. Fetch the latest results
        kb_results = api_results['kb']
        sorted_kb_results = sorted(
            kb_results, key=lambda x: -int(x['R_rating'])
        )

        new_utterances = self.rewrite(dialog.utterances, sorted_kb_results)

        random.shuffle(kb_results)
        api_results['kb'] = self.parse_results(deepcopy(kb_results))

        dialog.update(
            utterances=new_utterances,
            api_results=api_results
        )

        return dialog

    @staticmethod
    def parse_results(records):
        """
        Convert KB results to list of tuples
        :param res_df:
        :return: [tuple] list of triplets sorted with rating
        """
        columns = [
            'R_phone',
            'R_cuisine',
            'R_address',
            'R_location',
            'R_number',
            'R_price',
            'R_rating',
        ]

        ret_tuples = []
        for record in records:
            rname = record['R_restro']
            for key in columns:
                ret_tuples.append((rname, key, record[key]))

        return ret_tuples


class DynamicDialogRewriter(BaseDialogRewriter):
    def __init__(self):
        """
        Dialog rewriter for randomized KB engine
        :param engine: KbEngine object
        """
        super(DynamicDialogRewriter, self).__init__()

    def rewrite(self, utterances, kb_results):
        new_utterances = deepcopy(utterances)
        base_idx = self.detect_rewrite_start(utterances)

        if len(kb_results) == 0:
            # No valid KB results
            new_utterances = new_utterances[:base_idx] +\
                generate_empty_results_utterances()
            return new_utterances

        # Got at least one result from kb
        kb_ptr = 0
        do_trim = False
        trim_start = 0

        for idx in range(base_idx, len(utterances)):
            user, uttr = utterances[idx]

            if user != 'bot':
                continue

            if re.search(next_option_regex, uttr) is not None:
                kb_ptr += 1
                if kb_ptr == len(kb_results):
                    # Means original query had more results than current. Trim
                    do_trim = True
                    trim_start = idx - 1
                    break

                continue

            new_uttr = self.update_bot_utterance(uttr, kb_results[kb_ptr])
            new_utterances[idx] = (user, new_uttr)

        if do_trim:
            # Basically just cut-off everything from last user
            # utterance till final suggestion
            def func(idx):
                return (
                    utterances[idx][0] == 'bot' and
                    re.search(suggestion_regex, utterances[idx][1]) is not None
                )

            idxs = list(filter(func, range(trim_start, len(utterances))))

            trim_end = idxs[-1] + 1
            new_utterances = new_utterances[:trim_start] +\
                new_utterances[trim_end:]

            for idx in range(trim_start, len(new_utterances)):
                user, uttr = new_utterances[idx]

                if user != 'bot':
                    continue

                new_uttr = self.update_bot_utterance(uttr, kb_results[-1])
                new_utterances[idx] = (user, new_uttr)

        return new_utterances
