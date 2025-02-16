# Details of the hybrid model switching conditions

T_parts = [-17.45, (-17.45 -6.55)/2,  -6.55, (-6.55 + 2.4)/2, 2.4, (2.4 + 15.5)/2, 15.5]
intervals = [(T_parts[i], T_parts[i+1]) for i in range(len(T_parts)-1)]
Nparts = len(intervals)

# A wider partition
T_parts_w = [-17.45, -6.55, 2.4, 15.5]
intervals_w = [(T_parts_w[i], T_parts_w[i+1]) for i in range(len(T_parts_w)-1)]
Nparts_w = len(intervals_w)

# No partition
T_parts_n = [-17.45, 15.5]
intervals_n = [(T_parts_n[i], T_parts_n[i+1]) for i in range(len(T_parts_n)-1)]
Nparts_n = len(intervals_n)


def get_interval_T(T: float, intervals: list = intervals) -> int:
    """ The intervals are treated as half-open on the higher side i.e., [a, b)"""
    for i in range(Nparts):
        if intervals[i][0] <= T <= intervals[i][1]:  # this returns the first interval it belongs to unless
            return i                                           # the last value


if __name__ == '__main__':
    Ti = 8.95
    inters = intervals
    interval_num = get_interval_T(Ti, inters)
    print("Intervals: ", inters)
    print("Interval of Ti={} is ".format(Ti) + str(interval_num) + ": " + str(inters[interval_num]))