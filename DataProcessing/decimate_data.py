import numpy as np
import decimation as dc
import filt_data as fd
import etaCalc


class decimatedTestData():
    """ The Class of decimated data """
    # ========================================================================
    def __init__(self, age: int, test_type: int):
        """ Initializes the data classes """
        self.filtData = fd.FilteredTestData(age, test_type)
        self.dt = 1
        self.name = self.filtData.name
        self.ssd = self.decimate_ssd()
        self.iod = self.decimate_iod()

    # ========================================================================
    def decimate_ssd(self) -> dict[str, np.ndarray]:
        """ Decimate the filtered ssd data """
        ssd_keys = ['x1', 'x2', 'u1', 'u2', 'F', 'T', 'eta']
        ssd = {}
        for key in ssd_keys:
            ssd[key] = dc.decimate_withTD(self.filtData.ssd['t_skips'], self.filtData.ssd[key])
        ssd['t'] = dc.decimate_time2OneHz(self.filtData.ssd['t_skips'], self.filtData.ssd['t'])
        ssd['t_skips'] = fd.find_discontinuities(ssd['t'], self.dt)
        ssd['eta_dec'] = ssd['eta']
        ssd['eta'] = etaCalc.calc_eta_TD(ssd['x1'], ssd['u1'], ssd['t_skips'])
        return ssd

    # =========================================================================
    def decimate_iod(self):
        """ Decimate the filtered iod data """
        iod_keys = ['y1', 'u1', 'u2', 'F', 'T', 'eta']
        iod = {}
        for key in iod_keys:
            iod[key] = dc.decimate_withTD(self.filtData.iod['t_skips'], self.filtData.iod[key])
        iod['t'] = dc.decimate_time2OneHz(self.filtData.iod['t_skips'], self.filtData.iod['t'])
        iod['t_skips'] = fd.find_discontinuities(iod['t'], self.dt)
        iod['eta_dec'] = iod['eta']
        iod['eta'] = etaCalc.calc_eta_TD(iod['y1'], iod['u1'], iod['t_skips'])
        return iod

# ======================================================================================================================

## =====================================================================================================================
def load_decimated_test_data_set():
    # Load the test Data
    decimated_test_data = [[decimatedTestData(age, tst) for tst in range(3)] for age in range(2)]
    return decimated_test_data

# ======================================================================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('tkAgg')

    # Actually load the entire Data set ----------------------------------------
    dct = load_decimated_test_data_set()
    fig_dpi = 300
    show_plot = 'u1'

    # Plotting all the Data sets
    for i in range(2):
        for j in range(3):
            for key in ['u1', 'u2', 'T', 'F', 'x1', 'x2', 'eta']:
                plt.figure()
                if (key != 'eta'):
                    plt.plot(dct[i][j].filtData.rawData.raw['t'], dct[i][j].filtData.rawData.raw[key], ':', label=key+"_raw", linewidth=1)
                plt.plot(dct[i][j].filtData.ssd['t'], dct[i][j].filtData.ssd[key], '-.', label= key+"_filtered", linewidth=1)
                plt.plot(dct[i][j].ssd['t'], dct[i][j].ssd[key], '--',label=key + "_decimated", linewidth=1)
                if (key == 'eta'):
                    plt.plot(dct[i][j].ssd['t'], dct[i][j].ssd['eta_dec'], '--', label="decimate(eta_filtered)", linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(dct[i][j].name + "_ssd")
                plt.savefig("figs/" + dct[i][j].name + "_ssd_" + key + ".png", dpi=fig_dpi)
                if key != show_plot:
                    plt.close()

            for key in ['u1', 'u2', 'T', 'F', 'y1', 'eta']:
                plt.figure()
                if (key != 'eta'):
                    plt.plot(dct[i][j].filtData.rawData.raw['t'], dct[i][j].filtData.rawData.raw[key], ':', label=key+"_raw", linewidth=1)
                plt.plot(dct[i][j].filtData.iod['t'], dct[i][j].filtData.iod[key], '-.', label= key+"_filtered", linewidth=1)
                plt.plot(dct[i][j].iod['t'], dct[i][j].iod[key], '--', label=key + "_decimated", linewidth=1)
                if (key == 'eta'):
                    plt.plot(dct[i][j].iod['t'], dct[i][j].iod['eta_dec'], '--', label="decimate(eta_filtered)", linewidth=1)
                plt.grid()
                plt.legend()
                plt.xlabel('Time [s]')
                plt.ylabel(key)
                plt.title(dct[i][j].name + "_iod")
                plt.savefig("figs/" + dct[i][j].name + "_iod_" + key + ".png", dpi=fig_dpi)
                if key != show_plot:
                    plt.close()

    # Showing datat discontinuities --------------------------------------------
    plt.figure()
    for i in range(2):
        for j in range(3):
            t = dct[i][j].ssd['t']
            plt.plot(np.arange(len(t)), t, '--', label=dct[i][j].name + 'ss', linewidth=1)
            t = dct[i][j].iod['t']
            plt.plot(np.arange(len(t)), t, '--', label=dct[i][j].name + 'io', linewidth=1)
    plt.grid()
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Time [s]')
    plt.title('Time discontinuities in test Data')
    plt.savefig("figs/time_discontinuities_test.png", dpi=fig_dpi)
    plt.close()

    plt.show()
    # plt.close('all')
