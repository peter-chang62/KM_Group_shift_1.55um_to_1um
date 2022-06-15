import numpy as np
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import matplotlib.pyplot as plt
import simulationHeader as sh
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()


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


# %%___________________________________________ plotting _______________________________________________________________
pulse = get_1p68nJ_pulse()
# pulse = get_2p02nJ_pulse()

POWER = []
Z = []
Length = np.linspace(7, 11, 50)
for n, length in enumerate(Length):
    sim_pm1550 = sh.simulate(pulse=pulse,
                             fiber=sh.fiber_pm1550,
                             length_cm=length,
                             epp_nJ=1.68,
                             nsteps=200)

    sim_adhnlf = sh.simulate(
        pulse=sim_pm1550.pulse,
        fiber=sh.fiber_adhnlf,
        length_cm=7,
        epp_nJ=sim_pm1550.pulse.calc_epp() * 1e9,
        nsteps=300
    )

    window = .020  # 20 nm
    power = fpn.power_in_window(pulse, sim_adhnlf.AW, 1 - window / 2, 1 + window / 2, 100)
    ind = np.argmax(power)
    AW = sim_adhnlf.AW[ind]
    zs_best = sim_adhnlf.zs[ind]
    sim_adhnlf.pulse.set_AW(AW)

    sim_pm15502 = sh.simulate(
        pulse=sim_adhnlf.pulse,
        fiber=sh.fiber_pm1550,
        length_cm=17,
        epp_nJ=sim_adhnlf.pulse.calc_epp() * 1e9,
        nsteps=200
    )

    power = fpn.power_in_window(pulse, sim_pm15502.AW, 1 - window / 2, 1 + window / 2, 100)[-1]

    POWER.append(power)
    Z.append(zs_best)

    print(f'_______________________________{len(Length) - n}____________________________________________________')

Z = np.array(Z)
POWER = np.array(POWER)

ARR = np.c_[Length, Z, POWER]
with open('res_pm1550_to_HNLF_to_pm1550.npy', 'wb') as f:
    np.save(f, ARR)

# %%___________________________________________ plotting _______________________________________________________________
fig, ax = plt.subplots(1, 1)
ax.plot(Length, Z, '.-', label='HNLF length')
ax2 = ax.twinx()
ax2.plot(Length, POWER * 1e3, '.-', label='Power at 1 $\mathrm{\mu m}$', color='C1')
ax.set_xlabel("PM1550 length (cm)")
ax.set_ylabel("HNLF Length (cm)")
ax2.legend(loc='best')
ax.legend(loc='best')
ax2.set_ylabel("Power (mW at 100 MHz)")

# %%___________________________________________ plotting _______________________________________________________________
# sh.plot_freq_evolv(sim_pm15502, xlims=[.8, 2])
# plt.axhline(zs_best, color='r')
# plt.plot(sim_adhnlf.zs * 100, power * 1e3)
# sh.plot_cross_section(sim_adhnlf, sim_adhnlf.zs[best] * 100, xlims=[.8, 2])
# plt.title(("%.1f" % np.max(power * 1e3)) + " mW")
