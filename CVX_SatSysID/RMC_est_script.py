import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from DataProcessing import decimate_data as dd
import cvxpy as cp

###
rmc = dd.decimatedTestData(0, 2);  # Degreened RMC Data.
deltas = np.abs(rmc.ssd['x1'] - rmc.iod['y1'])
