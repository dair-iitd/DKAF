import numpy as np
from datetime import datetime, timedelta


class BaseTimeStamp(object):
    def __init__(self, ts):
        self.ts = ts


class BaseClock(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.step_size = cfg['step_size']
        self.horizon = cfg['horizon']
        self.curr_time = 0

    def get_clock_start(self):
        return BaseTimeStamp(0)

    def tick(self):
        if self.curr_time >= self.horizon:
            return None

        event = BaseTimeStamp(self.curr_time)
        self.curr_time += self.step_size

        return event


class CalendarClock(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # all dates in yyyy-mm-dd format
        date = cfg['start_date']
        self.start_date = datetime.strptime(date, '%Y-%m-%d')
        date = cfg['end_date']
        self.end_date = datetime.strptime(date, '%Y-%m-%d')
        assert self.start_date < self.end_date, 'Start date > End date'

        # Step size details
        step_size = cfg['step_size']  # in minutes 1-60
        # assert step_size in range(1, 61), 'Incorrect Step Size'
        self.step_size = timedelta(minutes=step_size)

        self.use_dayname = self.cfg.get('use_dayname', True)

        self.curr_time = self.start_date

    def reset(self):
        self.curr_time = self.start_date

    def datetime_to_dict(self, time):
        ts = time.time()
        stamp = int(ts.hour * 100 + ts.minute)
        weekday = time.strftime("%A") if self.use_dayname else time.weekday()

        return {
            'year': time.year,
            'month': time.month,
            'day': time.day,
            'hours': ts.hour,
            'minutes': ts.minute,
            'time_of_the_day': stamp,
            'weekday': weekday,
            'timestamp': time.timestamp(),
        }

    def get_clock_start(self):
        return self.datetime_to_dict(self.start_date)

    def tick(self):
        if self.curr_time >= self.end_date:
            return None

        event = self.curr_time
        self.curr_time = self.curr_time + self.step_size

        return self.datetime_to_dict(event)


# Tests start here!
def test_base_clock():
    step_size = 15
    horizon = 105
    cfg = {
        'step_size': step_size,
        'horizon': horizon,
    }
    clock = BaseClock(cfg)

    ecnt = 0
    event = clock.tick()
    while event is not None:
        ecnt += 1
        print(f'Event time {event.ts}')
        event = clock.tick()

    exp_ecnt = int(np.ceil((1.0 * horizon) / step_size))
    if exp_ecnt != ecnt:
        print(
            'FAILURE: '
            f'Actual event count {ecnt} does not '
            f'match expected count {exp_ecnt}'
        )
    else:
        print('SUCCESS!')


def test_calendar_clock():
    step_size = 30
    start_date = "2020-01-01"
    end_date = "2020-01-02"
    cfg = {
        'step_size': step_size,
        'start_date': start_date,
        'end_date': end_date,
    }
    clock = CalendarClock(cfg)

    ecnt = 0
    event = clock.tick()
    while event is not None:
        ecnt += 1
        print(f'Event time {event}')
        event = clock.tick()

    exp_ecnt = 24 * 2
    if exp_ecnt != ecnt:
        print(
            'FAILURE: '
            f'Actual event count {ecnt} does not '
            f'match expected count {exp_ecnt}'
        )
    else:
        print('SUCCESS!')


if __name__ == "__main__":
    test_base_clock()
    test_calendar_clock()
