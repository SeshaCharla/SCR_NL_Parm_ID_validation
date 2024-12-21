import numpy as np
import rdRawDat as rd
import cdRLS_smoothing as cdRLS
import scipy.signal as sig
import sosFiltering as sf
import etaCalc

# Array Manipulating functions ------------------------------------------------------
#==============================================================================================
def find_discontinuities(t, dt):
    """Find the discontinuities in the time Data
    The slices would be: [[t_skips[0], t_skips[1]], ...
    """
    t_skips = np.array([i for i in range(1, len(t))
                        if t[i] - t[i - 1] > 1.5 * dt], dtype=int)
    t_skips = np.append(t_skips, len(t))
    t_skips = np.insert(t_skips, 0, 0)
    return t_skips


# =============================================================================================
def rmNaNrows(x):
    """Remove the rows with NaN values"""
    return np.delete(x,
                     [i for i in range(len(x))
                         if np.any(np.isnan(x[i]))],
                     axis=0)


#===============================================================================================
class FilteredTestData():
    """Class of filtered test data both ssd and iod"""
    #===========================================================================================
    def __init__(self, age: int, test_type: int):
        self.rawData = rd.RawTestData(age, test_type)
        self.dt = self.rawData.dt
        self.name = self.rawData.name
        self.cdRLS_parms = cdRLS.cdRLS_parms("test")
        self.ssd = self.gen_ssd()
        self.iod = self.gen_iod()

    # ==========================================================================================
    def gen_ssd(self) -> dict[str, np.ndarray]:
        # Generate the state space Data
        raw_tab = np.matrix([self.rawData.raw['t'],
                             self.rawData.raw['x1'],
                             self.rawData.raw['x2'],
                             self.rawData.raw['u1'],
                             self.rawData.raw['u2'],
                             self.rawData.raw['T'],
                             self.rawData.raw['F']]).T
        ssd_tab = rmNaNrows(raw_tab)
        ssd_mat = ssd_tab.T
        ssd = {}
        ssd['t'] = np.array(ssd_mat[0]).flatten()
        ssd['x1'] = np.array(ssd_mat[1]).flatten()
        ssd['x2'] = np.array(ssd_mat[2]).flatten()
        ssd['u1'] = np.array(ssd_mat[3]).flatten()
        ssd['u2'] = np.array(ssd_mat[4]).flatten()
        ssd['T'] = np.array(ssd_mat[5]).flatten()
        ssd['F'] = np.array(ssd_mat[6]).flatten()
        # Find the time discontinuities in SSD Data
        ssd['t_skips'] = find_discontinuities(ssd['t'], self.dt)
        # Smooth all the data
        for state in ['x1', 'x2', 'u1', 'u2', 'T', 'F']:
            ssd[state] = sf.sosff_TD(ssd['t_skips'], ssd[state])
        # Calculating eta
        ssd['eta'] = etaCalc.calc_eta_TD(ssd['x1'], ssd['u1'], ssd['t_skips'])
        return  ssd

    # ===========================================================================================
    def gen_iod(self) -> dict[str, np.ndarray]:
        # Generate the input output Data
        raw_tab = np.matrix([self.rawData.raw['t'],
                             self.rawData.raw['y1'],
                             self.rawData.raw['u1'],
                             self.rawData.raw['u2'],
                             self.rawData.raw['T'],
                             self.rawData.raw['F']]).T
        iod_tab = rmNaNrows(raw_tab)
        # Clearing non-existant iod data, y1 doesn't work bellow a certain temperature
        if self.name in ["dg_cftp", "aged_cftp"]:
            print("clearing non-existant y1 in " + self.name + " data")
            iod_tab = np.copy(iod_tab[int(950/self.dt):])
        elif self.name in ["dg_hftp", "aged_hftp"]:
            print("clearing non-existant y1 in " + self.name + " data")
            iod_tab = np.copy(iod_tab[int(400/self.dt):int(600/self.dt)])
        iod_mat = iod_tab.T
        iod = {}
        iod['t'] = np.array(iod_mat[0]).flatten()
        iod['y1'] = np.array(iod_mat[1]).flatten()
        iod['u1'] = np.array(iod_mat[2]).flatten()
        iod['u2'] = np.array(iod_mat[3]).flatten()
        iod['T'] = np.array(iod_mat[4]).flatten()
        iod['F'] = np.array(iod_mat[5]).flatten()
        # Find the time discontinuities in IOD Data
        iod['t_skips'] = find_discontinuities(iod['t'], self.dt)
        # Smooth all the data
        for state in ['y1', 'u1', 'u2', 'T', 'F']:
            iod[state]= sf.sosff_TD(iod['t_skips'], iod[state])
        # Calculate eta
        iod['eta'] = etaCalc.calc_eta_TD(iod['y1'], iod['u1'], iod['t_skips'])
        return iod


# ======================================================================================================================

## =====================================================================================================================
def load_filtered_test_data_set():
    # Load the test Data
    filtered_test_data = [[FilteredTestData(age, tst) for tst in range(3)] for age in range(2)]
    return filtered_test_data

# ======================================================================================================================


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Actually load the entire Data set ----------------------------------------
    test_data = rd.load_test_data_set()
    filtered_test_data = load_filtered_test_data_set()
    fig_dpi = 300

    # Plotting all the Data sets
    for i in range(2):
        for j in range(3):
            for key in ['u1', 'u2', 'T', 'F', 'x1', 'x2', 'eta']:
                plt.figure()
                if (key != 'eta'):
                    plt.plot(test_data[i][j].raw['t'], test_data[i][j].raw[key], '--', label=key, linewidth=1)
                plt.plot(filtered_test_data[i][j].ssd['t'], filtered_test_data[i][j].ssd[key], label= key+"_filtered", linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(test_data[i][j].name)
                plt.savefig("figs/" + filtered_test_data[i][j].name + "_ssd_" + key + ".png", dpi=fig_dpi)
                plt.close()
            for key in ['u1', 'u2', 'T', 'F', 'y1', 'eta']:
                plt.figure()
                if (key != 'eta'):
                    plt.plot(test_data[i][j].raw['t'], test_data[i][j].raw[key], '--', label=key, linewidth=1)
                plt.plot(filtered_test_data[i][j].iod['t'], filtered_test_data[i][j].iod[key], label=key + "_filtered", linewidth=1)
                if (key == 'y1'):
                    plt.plot(filtered_test_data[i][j].ssd['t'], filtered_test_data[i][j].ssd['x1'], '--', label='x1_filtered', linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(test_data[i][j].name)
                plt.savefig("figs/" + test_data[i][j].name + "_iod_" + key + ".png", dpi=fig_dpi)
                plt.close()

    # Showing datat discontinuities --------------------------------------------
    plt.figure()
    for i in range(2):
        for j in range(3):
            t = filtered_test_data[i][j].ssd['t']
            plt.plot(np.arange(len(t)), t, label=test_data[i][j].name + 'ss', linewidth=1)
            t = filtered_test_data[i][j].iod['t']
            plt.plot(np.arange(len(t)), t, label=test_data[i][j].name + 'io', linewidth=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in test Data')
    plt.savefig("figs/time_discontinuities_test.png", dpi=fig_dpi)
    plt.close()

    # plt.show()
    plt.close('all')




## Not usefull sutff
"""Remove the data from IOD tab where the temperature is less than 200 + y_Tmin deg C"""
"""
# ==================================================================================================================
    def IOD_temp_exclusion(self, iod_tab):
        y_Tmin = 20
        return np.delete(iod_tab,
                         [i for i in range(len(iod_tab))
                                        if (self.rawData.raw['T'])[i]< y_Tmin],
                         axis=0)

"""