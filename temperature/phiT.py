import  numpy as np

T_ord = dict()
T_ord['ads'] = 1
T_ord['od'] = 1
T_ord['scr'] = 1
T_ord['Gamma'] = 2


def phi_T(T: float, ord: int) -> np.ndarray:
    """ Returns phi(T) for the given polynomial order"""
    T_poly = [T**n for n in range(ord, -1, -1)]
    phiT = np.matrix(T_poly).T
    return phiT

#===
if __name__ == "__main__":
    import pprint as pp
    pp.pprint(T_ord)
    print(phi_T(5, T_ord['Gamma']))