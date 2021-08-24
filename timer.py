# time measure

import datetime

class TimeMeasure:

    cycle_cnt = 0 # all checkpoints are run in cycle; this is cycle counter
    first_ckp = None # label of first ckp in cycle (to count cycles)
    last_ckp = None  # last checkpoints called
    checkpoints = {} # info about all checkpoints$ they will be added by .set()
    measure_needed = True

    @staticmethod
    def set(label):

        self = TimeMeasure
        if self.measure_needed == False: return

        if self.cycle_cnt == 0:
            self.first_ckp = label

        if label not in self.checkpoints.keys(): # new checkpoint
            self.checkpoints[label]={
                'prev_ckp': self.last_ckp,
                'total_interval_time': (datetime.datetime.now() - datetime.datetime.now() )
            }

        now = datetime.datetime.now()

        my_ckp = self.checkpoints[label]

        if label == self.first_ckp:  # first ckp in cycle
            my_ckp['prev_ckp'] = self.last_ckp  # connect cycle
            self.cycle_cnt += 1

        prev_label = my_ckp['prev_ckp']
        if prev_label is not None: # first ckp, prev hasn't established yet
            if prev_label != self.last_ckp:
                print('TimeMeasure error: more than one path of checkpoints!!  prev_ckp={} while last_ckp={}'
                                                                                .format(prev_label, self.last_ckp))
            prev_ckp = self.checkpoints[ prev_label ]
            prev_label_time_mark = prev_ckp[ 'last_time_mark' ]
            my_ckp['total_interval_time'] += (now - prev_label_time_mark)
        my_ckp['last_time_mark'] = now
        self.last_ckp = label

    @staticmethod
    def results():

        tm = TimeMeasure
        if not tm.measure_needed : return ''
        res_str = ''

        total_sec = 0.0
        for ckp in tm.checkpoints:
            if ckp == "skip_me":
                continue  # don't include in statistics
            total_sec += (tm.checkpoints[ckp]['total_interval_time']).total_seconds()
        beg_str = f'\nTime measure total:   cycles = {tm.cycle_cnt}'
        res_str += f'{beg_str}   seconds = {total_sec:8.2f},   msec/cycle = {(total_sec/tm.cycle_cnt)*1000 :4.0f}  \
            cycles/sec = {tm.cycle_cnt/total_sec:4.0f}\n'

        for ckp in tm.checkpoints:
            if ckp == "skip_me":
                continue  # don't include in statistics
            total_interv_sec = (tm.checkpoints[ckp]['total_interval_time']).total_seconds()
            interv_labels = f"{tm.checkpoints[ckp]['prev_ckp']:35s} - {ckp:35s}"
            res_str += '{:30s}: seconds = {:8.2f},  ms/cycle = {:4.0f} ({:4.1f}%)\n'.format(
                interv_labels, total_interv_sec,
                (total_interv_sec/tm.cycle_cnt)*1000, (total_interv_sec/total_sec)*100 )
        return res_str

if __name__ == '__main__':

    def timeload():
        s='0'
        for i in range(100):
            s+= '*'

    print ('Time measure test')
    tm = TimeMeasure
    for i in range(100):
        tm.set('label_1')
        for j in range(1000): timeload()

        tm.set('label_2')
        for j in range(10000): timeload()

        tm.set('label_3')
        for j in range(300): timeload()

    print('\n \nresults:\n{}'.format(tm.results()))