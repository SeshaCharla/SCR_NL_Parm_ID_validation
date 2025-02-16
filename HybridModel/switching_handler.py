# Details of the hybrid model switching conditions

T_parts = [-17.45, (-17.45 -6.55)/2,  -6.55, (-6.55 + 2.4)/2, 2.4, (2.4 + 15.5)/2, 15.5]
intervals = [(T_parts[i], T_parts[i+1]) for i in range(len(T_parts)-1)]
Nparts = len(intervals)

def get_interval_T(T: float) -> int:
    """ The intervals are treated as half-open on the higher side i.e., [a, b)"""
    for i in range(Nparts):
        if intervals[i][0] <= T <= intervals[i][1]:  # this returns the first interval it belongs to unless
            return i                                           # the last value


if __name__ == '__main__':
    print("Intervals: ", intervals)
    Ti = 8.95
    print("Interval of Ti={} is ".format(Ti) + str(get_interval_T(Ti)) + ": " + str(intervals[get_interval_T(Ti)]))