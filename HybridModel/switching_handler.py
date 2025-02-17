import numpy as np

# Details of the hybrid model switching conditions
T_parts = [-17.45, (-17.45 -6.55)/2,  -6.55, (-6.55 + 2.4)/2, 2.4, (2.4 + 15.5)/2, 15.5]
# A wider partition
T_parts_w = [-17.45, -6.55, 2.4, 15.5]
# No Partitions
T_parts_n = [-17.45, 15.5]
# By default T_parts is used. We need to replace with a wider partions or no partition, T_parts needed to be reassigned.
T_parts = T_parts_w

intervals = [(T_parts[i], T_parts[i+1]) for i in range(len(T_parts)-1)]
Nparts = len(intervals)
part_keys = [str(np.array(intervals[i]) * 10 + 200) for i in range(Nparts)]


def get_interval_T(T: float) -> int:
    """ The intervals are treated as half-open on the higher side i.e., [a, b)"""
    for i in range(Nparts):
        if intervals[i][0] <= T <= intervals[i][1]:  # this returns the first interval it belongs to unless
            return i                                           # the last value


if __name__ == '__main__':
    Ti = 8.95
    interval_num = get_interval_T(Ti)
    print("Intervals: ", intervals)
    print("Interval of Ti={} is ".format(Ti) + str(interval_num) + ": " + str(intervals[interval_num]))