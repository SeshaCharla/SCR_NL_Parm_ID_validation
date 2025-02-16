import  numpy as np


def phi_T(T: float, ord: int) -> np.ndarray:
    """ Returns phi(T) for the given polynomial order"""
    T_poly = [T**n for n in range(ord, -1, -1)]
    phiT = np.matrix(T_poly).T
    return phiT

#===
if __name__ == "__main__":
    print(phi_T(5, 2))