import numpy as np

# Temperature Ranges
T_narrow = [-17.45, (-17.45 -6.55)/2,  -6.55, (-6.55 + 2.4)/2, 2.4, (2.4 + 15.5)/2, 15.5]
T_wide = [-17.45, -6.55, 2.4, 15.5]
T_none = [-17.45, 15.5]
T_hl = [-17.45, -6.55, 15.5]

class switch_handle:
    """ Class to handle switching based on temperature """
    def __init__(self, T_parts: list):
        """ Initialize the switching handler """
        self.T_parts = T_parts
        self.intervals = [(T_parts[i], T_parts[i+1]) for i in range(len(T_parts)-1)]
        self.Nparts = len(self.intervals)
        self.part_keys = [str(np.array(self.intervals[i]) * 10 + 200) for i in range(self.Nparts)]
    # =============================================================================================

    def get_interval_T(self, T: float) -> str:
        """ The intervals are treated as half-open on the higher side i.e., [a, b)"""
        for i in range(self.Nparts):
            if self.intervals[i][0] <= T <= self.intervals[i][1]:  # this returns the first interval it belongs to unless
                return self.part_keys[i]                                           # the last value


if __name__ == '__main__':
    sh = switch_handle(T_narrow)
    Ti = 8.95
    interval_num = sh.get_interval_T(Ti)
    print("Intervals: ", sh.intervals)
    print("Interval of Ti={} is ".format(Ti) + str(interval_num))
