from PyEMD import EEMD
import numpy as np
import pylab as plt
from cwru.core import cwru_input


def eemd(signal):
    """
    eemd分解
    :param signal:采样信号
    :return:
    """
    # Assign EEMD to `eemd` variable
    eemd = EEMD()
    # Say we want detect extrema using parabolic method
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    # Execute EEMD on S
    eIMFs = eemd.eemd(S)
    return eIMFs


if __name__ == "__main__":
    # Define signal
    data = cwru_input.read_matdata('../data/12k_Drive_End_B007_0_118.mat')
    print(data.shape)
    S = data[0:400]
    t = np.linspace(0, 1, 400)
    eIMFs = eemd(S)
    nIMFs = eIMFs.shape[0]
    # Plot results
    plt.figure(figsize=(12, 9))
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(t, S, 'r')

    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('105', dpi=120)
    plt.show()