import numpy as np
from pandas import read_csv
import unit_convs as uc


class RawTestData():
    """Class that reads the raw test data
     Does the unit conversions"""

    # ==============================================================
    def __init__(self, age: int, test_type: int):
        """Reads the test data and stores into a .raw dictionary"""
        self.dt = 0.2           # Sampling time
        self.name = self.test_name(age, test_type)
        self.dat_file = self.data_dir()
        self.raw = self.load_test_data()

    # ==============================================================
    def load_test_data(self) -> dict[str, np.ndarray]:
        """Loads the test data"""
        data = read_csv(self.dat_file, header=[0, 1])
        raw_data = {}
        # ======================================================================================
        # Assigning the Data to the variables
        # Time is in seconds
        raw_data['t'] = np.array(data.get(('LOG_TM', 'sec')), dtype=np.float64).flatten()
        # ======================================================================================
        # Mass flow rate is in g/sec
        F_kgmin = np.array(data.get(('EXHAUST_FLOW', 'kg/min')), dtype=np.float64).flatten()
        raw_data['F'] = uc.uConv(F_kgmin,"kg/min to g/s")        # g/sec
        # ======================================================================================
        # Temperature is in deg-C
        Tin = np.array(data.get(('V_AIM_TRC_DPF_OUT', 'Deg_C')), dtype=np.float64).flatten()
        Tout = np.array(data.get(('V_AIM_TRC_SCR_OUT', 'Deg_C')), dtype=np.float64).flatten()
        Tscr = np.mean([Tin, Tout], axis=0).flatten()
        raw_data['T'] = uc.uConv(Tscr, "-T0C")
        # =======================================================================================
        # NOx output is in mol/m^3
        NOx = np.array(data.get(('EXH_CW_NOX_COR_U1', 'PPM')), dtype=np.float64).flatten()
        raw_data['x1'] = uc.uConv(NOx, "NOx ppm to mol/m^3")
        # =======================================================================================
        # NH3 output is in mol/m^3
        NH3 = np.array(data.get(('EXH_CW_AMMONIA_MEA', 'ppm')), dtype=np.float64).flatten()
        raw_data['x2'] = uc.uConv(NH3, "NH3 ppm to mol/m^3")
        # =======================================================================================
        # NOx out measured in mol/m^3
        y1 = np.array(data.get(('V_SCM_PPM_SCR_OUT_NOX', 'ppm')), dtype=np.float64).flatten()
        raw_data['y1'] = uc.uConv(y1, "NOx ppm to mol/m^3")
        # =======================================================================================
        # NOx input is in mol/m^3
        u1 = np.array(data.get(('ENG_CW_NOX_FTIR_COR_U2', 'PPM')), dtype=np.float64).flatten()
        raw_data['u1'] = uc.uConv(u1, "NOx ppm to mol/m^3")
        # ========================================================================================
        # Urea injection rate is in ml/sec
        raw_data['u2'] = np.array(data.get(('V_UIM_FLM_ESTUREAINJRATE', 'ml/sec')), dtype=np.float64).flatten()
        # u1_sensor = np.array(Data.get(('EONOX_COMP_VALUE', 'ppm'))).flatten()
        # ======================================================================================================
        return raw_data

    # ==============================================================
    def test_name(self, age: int, test_type: int) -> str:
        """ Data names for the truck and test Data
        [0][j] - Degreened Data
        [1][j] - Aged Data
        """
        test = [["dg_cftp", "dg_hftp", "dg_rmc"],
                ["aged_cftp", "aged_hftp", "aged_rmc"]]
        return test[age][test_type]

    # ==============================================================
    def data_dir(self) -> str:
        """Returns the data directory for the test data"""
        dir_prefix = "../../Data"
        test_dir_prefix = dir_prefix + "/test_cell_data/"
        test_dict = {"aged_cftp": "g580040_Aged_cFTP.csv",
                     "aged_hftp": "g580041_Aged_hFTP.csv",
                      "aged_rmc": "g580043_Aged_RMC.csv",
                       "dg_cftp": "g577670_DG_cFTP.csv",
                       "dg_hftp": "g577671_DG_hFTP.csv",
                        "dg_rmc": "g577673_DG_RMC.csv"}
        return test_dir_prefix + test_dict[self.name]

    # ===============================================================