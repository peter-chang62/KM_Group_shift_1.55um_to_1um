import numpy as np
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import matplotlib.pyplot as plt
import simulationHeader as sh


def get_1p68nJ_pulse(plot=False):
    intensity = np.genfromtxt("KM_FROG_RETRIEVAL/1p68nJ_30p2cmpatch_Intensity.csv", delimiter=',')
    spectrum = np.genfromtxt("KM_FROG_RETRIEVAL/1p68nJ_30p2cmpatch_Spectrum.csv", delimiter=',')

    pulse = fpn.Pulse()
    amp = spectrum[:, 1].astype(np.complex128) ** 0.5
    phase = spectrum[:, 2]  # rad
    amp *= np.exp(1j * phase)
    pulse.set_AW_experiment(spectrum[:, 0] * 1e-3, amp)
    pulse.set_epp(1.68e-9)

    if plot:
        plt.figure()
        plt.plot(pulse.wl_um * 1e3, sh.normalize(pulse.AW.__abs__() ** 2))
        plt.plot(spectrum[:, 0], sh.normalize(spectrum[:, 1]))
        plt.xlim(1400, 1800)

        plt.figure()
        plt.plot(pulse.T_ps, sh.normalize(pulse.AT.__abs__() ** 2))
        plt.plot(intensity[:, 0] * 1e12, sh.normalize(intensity[:, 1]))

    return pulse


def get_2p02nJ_pulse(plot=False):
    intensity = np.genfromtxt("KM_FROG_RETRIEVAL/2p02nJ_11cmpatch_Intensity.csv", delimiter=',')
    spectrum = np.genfromtxt("KM_FROG_RETRIEVAL/2p02nJ_11cmpatch_Spectrum.csv", delimiter=',')

    pulse = fpn.Pulse()
    amp = spectrum[:, 1].astype(np.complex128) ** 0.5
    phase = spectrum[:, 2]  # rad
    amp *= np.exp(1j * phase)
    pulse.set_AW_experiment(spectrum[:, 0] * 1e-3, amp)
    pulse.set_epp(2.02e-9)

    if plot:
        plt.figure()
        plt.plot(pulse.wl_um * 1e3, sh.normalize(pulse.AW.__abs__() ** 2))
        plt.plot(spectrum[:, 0], sh.normalize(spectrum[:, 1]))
        plt.xlim(1400, 1800)

        plt.figure()
        plt.plot(pulse.T_ps, sh.normalize(pulse.AT.__abs__() ** 2))
        plt.plot(intensity[:, 0] * 1e12, sh.normalize(intensity[:, 1]))
    return pulse


pulse = get_1p68nJ_pulse()
# pulse = get_2p02nJ_pulse()

sim_pm1550 = sh.simulate(pulse=pulse,
                         fiber=sh.fiber_pm1550,
                         length_cm=7,
                         epp_nJ=1.68,
                         # epp_nJ=2.02,
                         nsteps=200)

sim_adhnlf = sh.simulate(
    pulse=sim_pm1550.pulse,
    # pulse=pulse,
    fiber=sh.fiber_adhnlf,
    # fiber=sh.fiber_adhnlf_2,
    length_cm=7,
    epp_nJ=sim_pm1550.pulse.calc_epp() * 1e9,
    # epp_nJ=1.68,
    # epp_nJ=2.02,
    nsteps=300
)

# %%___________________________________________ plotting _______________________________________________________________
window = .020  # 20 nm
power = fpn.power_in_window(pulse, sim_adhnlf.AW, 1 - window / 2, 1 + window / 2, 100)
best = np.argmax(power)

sh.plot_freq_evolv(sim_adhnlf, xlims=[.8, 2])
plt.axhline(sim_adhnlf.zs[best] * 100, color='r')
plt.plot(sim_adhnlf.zs * 100, power * 1e3)
sh.plot_cross_section(sim_adhnlf, sim_adhnlf.zs[best] * 100, xlims=[.8, 2])
plt.title(("%.1f" % np.max(power * 1e3)) + " mW")
