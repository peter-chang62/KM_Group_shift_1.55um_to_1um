import matplotlib.pyplot as plt
import numpy as np
import pynlo_peter.Fiber_PPLN_NLSE as fpn


def normalize(vec):
    return vec / np.max(abs(vec))


# %%____________________________________________________________________________________________________________________
intensity = np.genfromtxt("KM_FROG_RETRIEVAL/1p68nJ_30p2cmpatch_Intensity.csv", delimiter=',')
spectrum = np.genfromtxt("KM_FROG_RETRIEVAL/1p68nJ_30p2cmpatch_Spectrum.csv", delimiter=',')

pulse = fpn.Pulse()
amp = spectrum[:, 1].astype(np.complex128) ** 0.5
phase = spectrum[:, 2]  # rad
amp *= np.exp(1j * phase)
pulse.set_AW_experiment(spectrum[:, 0] * 1e-3, amp)

# %%____________________________________________________________________________________________________________________
# checks out!
plt.figure()
plt.plot(pulse.wl_um * 1e3, normalize(pulse.AW.__abs__() ** 2))
plt.plot(spectrum[:, 0], normalize(spectrum[:, 1]))
plt.xlim(1400, 1800)

plt.figure()
plt.plot(pulse.T_ps, normalize(pulse.AT.__abs__() ** 2))
plt.plot(intensity[:, 0] * 1e12, normalize(intensity[:, 1]))

# %%____________________________________________________________________________________________________________________
pulse.set_epp(1.68e-9)
