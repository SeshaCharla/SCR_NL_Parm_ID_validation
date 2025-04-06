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
        # Temperature is in deg-C
        Tin = np.array(data.get(('V_AIM_TRC_DPF_OUT', 'Deg_C')), dtype=np.float64).flatten()
        Tout = np.array(data.get(('V_AIM_TRC_SCR_OUT', 'Deg_C')), dtype=np.float64).flatten()
        Tscr = np.mean([Tin, Tout], axis=0).flatten()
        raw_data['T'] = uc.uConv(Tscr, Tscr, "-T0C")
        # ======================================================================================
        # Mass flow rate is in g/sec
        F_kgmin = np.array(data.get(('EXHAUST_FLOW', 'kg/min')), dtype=np.float64).flatten()
        raw_data['F'] = uc.uConv(F_kgmin,Tscr, "kg/min to 10 g/s")        # g/sec
        # =======================================================================================
        # NOx output is in mol/m^3
        NOx = np.array(data.get(('EXH_CW_NOX_COR_U1', 'PPM')), dtype=np.float64).flatten()
        raw_data['x1'] = uc.uConv(NOx, Tscr, "ppm to 10^-3 mol/m^3")
        # =======================================================================================
        # NH3 output is in mol/m^3
        NH3 = np.array(data.get(('EXH_CW_AMMONIA_MEA', 'ppm')), dtype=np.float64).flatten()
        raw_data['x2'] = uc.uConv(NH3, Tscr, "ppm to 10^-3 mol/m^3")
        # =======================================================================================
        # NOx out measured in mol/m^3
        y1 = np.array(data.get(('V_SCM_PPM_SCR_OUT_NOX', 'ppm')), dtype=np.float64).flatten()
        raw_data['y1'] = uc.uConv(y1, Tscr, "ppm to 10^-3 mol/m^3")
        # =======================================================================================
        # NOx input is in mol/m^3
        u1 = np.array(data.get(('ENG_CW_NOX_FTIR_COR_U2', 'PPM')), dtype=np.float64).flatten()
        raw_data['u1'] = uc.uConv(u1, Tscr, "ppm to 10^-3 mol/m^3")
        # ========================================================================================
        # Urea injection rate is in ml/sec
        u2 = np.array(data.get(('V_UIM_FLM_ESTUREAINJRATE', 'ml/sec')), dtype=np.float64).flatten()
        raw_data['u2'] = uc.uConv(u2, Tscr, "ml/s to 10^-1 ml/s")
        # u1_sensor = np.array(Data.get(('EONOX_COMP_VALUE', 'ppm'))).flatten()
        # ======================================================================================================
        return raw_data

    # ==============================================================
    def test_name(self, age: int, test_type: int) -> str:
        """ Data names for the truck and test Data
        [0][j] - Degreened Data
        [1][j] - Aged Data
        """
        test = [["dg_cftp", "dg_hftp", "dg_rmc",
                 "dg_cftp_1", "dg_hftp_1", "dg_rmc_1",
                 "dg_cftp_2", "dg_hftp_2", "dg_rmc_2",
                 "dg_cftp_3", "dg_hftp_3", "dg_rmc_3"],
                ["aged_cftp", "aged_hftp", "aged_rmc"]]
        return test[age][test_type]

    # ==============================================================
    def data_dir(self) -> str:
        """Returns the data directory for the test data"""
        dir_prefix = "../../Data"
        test_dir_prefix = "/test_cell_data/"
        add_test_dir_prefix = "/Additional_TC_Data_DG/"
        test_dict = {"aged_cftp": test_dir_prefix + "g580040_Aged_cFTP.csv",
                     "aged_hftp": test_dir_prefix + "g580041_Aged_hFTP.csv",
                      "aged_rmc": test_dir_prefix + "g580043_Aged_RMC.csv",
                       "dg_cftp": test_dir_prefix + "g577670_DG_cFTP.csv",
                       "dg_hftp": test_dir_prefix + "g577671_DG_hFTP.csv",
                        "dg_rmc": test_dir_prefix + "g577673_DG_RMC.csv",
                     "dg_cftp_1": add_test_dir_prefix + "FTP1/" + "g611150_cFTP.csv",
                     "dg_hftp_1": add_test_dir_prefix + "FTP1/" + "g611151_hFTP.csv",
                      "dg_rmc_1": add_test_dir_prefix + "RMC/"  + "RMCSET_400HP_020518.csv",
                     "dg_cftp_2": add_test_dir_prefix + "FTP2/" + "g598050_cFTP.csv",
                     "dg_hftp_2": add_test_dir_prefix + "FTP2/" + "g598051_hFTP.csv",
                      "dg_rmc_2": add_test_dir_prefix + "RMC/"  + "RMCSET_400HP_022618.csv",
                     "dg_cftp_3": add_test_dir_prefix + "FTP3/" + "g598940_cFTP.csv",
                     "dg_hftp_3": add_test_dir_prefix + "FTP3/" + "g598941_hFTP.csv",
                      "dg_rmc_3": add_test_dir_prefix + "RMC/"  + "RMCSET_400HP_040218.csv"
        }
        return dir_prefix + test_dict[self.name]

    # ===============================================================

## =====================================================================================================================
def load_test_data_set():
    # Load the test Data
    test_data = [[RawTestData(age, tst) for tst in range(3)] for age in range(2)]
    return test_data

# ======================================================================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Actually load the entire Data set ----------------------------------------
    test_data = load_test_data_set()
    fig_dpi = 300

    # Plotting all the Data sets
    for i in range(2):
        for j in range(3):
            for key in ['u1', 'u2', 'T', 'F', 'x1', 'x2', 'y1']:
                plt.figure()
                plt.plot(test_data[i][j].raw['t'], test_data[i][j].raw[key], label=test_data[i][j].name + " " + key, linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(test_data[i][j].name)
                plt.savefig("figs/" + test_data[i][j].name + "_raw_" + key + ".png", dpi=fig_dpi)
                plt.close()
