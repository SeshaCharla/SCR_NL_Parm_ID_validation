import numpy as np
import rdRawDat as rd
import cdRLS_smoothing as cdRLS
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
    return np.delete(x, [i for i in range(len(x))
                         if np.any(np.isnan(x[i]))], axis=0)


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
            ssd[state], g1, g2 = cdRLS.cdRLS_withTD(ssd['t_skips'], ssd[state],
                                                         self.cdRLS_parms.lmbda,
                                                         self.cdRLS_parms.nu[state],
                                                         self.cdRLS_parms.h[state])
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
            print("clearing non-existant " + self.name + " data")
            iod_tab = np.copy(iod_tab[int(950/self.dt):])
        elif self.name in ["dg_hftp", "aged_hftp"]:
            print("clearing non-existant " + self.name + " data")
            iod_tab = np.copy(iod_tab[int(500/self.dt):])
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
            iod[state], g1, g2 = cdRLS.cdRLS_withTD(iod['t_skips'], iod[state],
                                                         self.cdRLS_parms.lmbda,
                                                         self.cdRLS_parms.nu[state],
                                                         self.cdRLS_parms.h[state])
        # Calculate eta
        iod['eta'] = etaCalc.calc_eta_TD(iod['y1'], iod['u1'], iod['t_skips'])
        return iod


