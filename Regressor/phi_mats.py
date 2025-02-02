import numpy as np
from DataProcessing import decimate_data as dd
import phi_algorithm as phi_k

class Phi_Y_Mats():
    """ Container for regression matrices Y and Phi for each of the hybrid states """