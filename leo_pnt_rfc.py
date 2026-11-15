"""
LEO-PNT RF compatibility simulator
==================================
Small Python toolkit to study the RF compatibility of a LEO-PNT signal-in-space
with existing GNSS/RNSS signals (GPS L1 C/A, Galileo E1, ...).

What it does
  1. Generates BPSK-R(n) and sine-phased BOC(m,n) signals and checks the
     analytic power spectral density (PSD) against a Welch estimate.
  2. Computes the autocorrelation function (acquisition / tracking shape).
  3. Computes the Spectral Separation Coefficient (SSC) between an interfering
     LEO-PNT signal and a victim GNSS signal, versus carrier-frequency offset
     (the offset includes the LEO Doppler shift).
  4. Builds a link budget from orbital geometry (altitude, elevation, slant
     range, free-space loss, Doppler) and gives the effective C/N0 degradation
     seen by a GNSS receiver on the ground.

Conventions
  * Chip rate fc = n * 1.023 MHz, sub-carrier frequency fs = m * 1.023 MHz.
  * PSDs are normalised to unit power inside the receiver front-end bandwidth.
  * Effective C/N0 (Betz):  (C/N0)_eff = C / (N0 + kappa * C_I)
    with kappa = SSC in Hz^-1 (equivalent to the I0,eff / C_I ratio).
  * All power / EIRP / noise values are ILLUSTRATIVE parameters, not mission data.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

F0 = 1.023e6            # Hz, GNSS reference frequency
C_LIGHT = 299_792_458.0
RE = 6_371e3            # m, mean Earth radius
MU = 3.986004418e14     # m^3/s^2


# --------------------------------------------------------------------------
# 1. Signals and PSD
# --------------------------------------------------------------------------
def psd_bpsk(f, n):
    """Unit-power PSD (Hz^-1) of BPSK-R(n), rectangular chips."""
    fc = n * F0
    return (1.0 / fc) * np.sinc(f / fc) ** 2


def psd_boc_sine(f, m, n):
    """Unit-power PSD (Hz^-1) of sine-phased BOC(m,n); needs k = 2m/n even."""
    fc, fs = n * F0, m * F0
    k = 2 * m / n
    if abs(k - round(k)) > 1e-9 or int(round(k)) % 2 != 0:
        raise ValueError("this closed form needs 2m/n to be an even integer")
    f = np.where(np.abs(f) < 1.0, 1.0, f)          # avoid 0/0 at f = 0
    return fc * (np.tan(np.pi * f / (2 * fs)) * np.sin(np.pi * f / fc) / (np.pi * f)) ** 2


def make_psd(kind, n, m=None):
    if kind == "bpsk":
        return lambda f: psd_bpsk(f, n)
    if kind == "boc":
        return lambda f: psd_boc_sine(f, m, n)
    raise ValueError(kind)


def generate_signal(kind, n, m=None, n_chips=20000, sps=16, seed=0):
    """Baseband chip stream: random +/-1 chips, rectangular pulse, optional square sub-carrier."""
    rng = np.random.default_rng(seed)
    chips = rng.choice([-1.0, 1.0], size=n_chips)
    x = np.repeat(chips, sps)
    if kind == "boc":
        half = int(round(sps * n / (2 * m)))          # samples per half sub-carrier period
        sub = np.tile(np.r_[np.ones(half), -np.ones(half)], int(np.ceil(len(x) / (2 * half))))[: len(x)]
        x = x * sub
    fs_samp = sps * n * F0
    return x, fs_samp


def band_normalise(psd, bw, npts=48001):
    f = np.linspace(-bw / 2, bw / 2, npts)
    p = np.trapezoid(psd(f), f)
    return lambda ff: psd(ff) / p


# --------------------------------------------------------------------------
# 2. Autocorrelation
# --------------------------------------------------------------------------
def autocorrelation(psd, bw, taus):
    f = np.linspace(-bw / 2, bw / 2, 96001)
    g = psd(f)
    return np.array([np.trapezoid(g * np.cos(2 * np.pi * f * t), f) for t in taus]) / np.trapezoid(g, f)


# --------------------------------------------------------------------------
# 3. Spectral separation coefficient
# --------------------------------------------------------------------------
def ssc(psd_victim, psd_interf, bw, df=0.0, npts=48001):
    """SSC (Hz^-1) = integral over the front-end band of G_v(f) * G_i(f - df)."""
    f = np.linspace(-bw / 2, bw / 2, npts)
    return np.trapezoid(psd_victim(f) * psd_interf(f - df), f)


# --------------------------------------------------------------------------
# 4. Geometry and link budget
# --------------------------------------------------------------------------
def slant_range(h, elev_deg):
    e = np.radians(elev_deg)
    r = RE + h
    return np.sqrt(r**2 - (RE * np.cos(e)) ** 2) - RE * np.sin(e)


def max_doppler(h, elev_deg, f_carrier):
    """Max Doppler (Hz) for a circular orbit in the user's plane, Earth rotation ignored."""
    r = RE + h
    v = np.sqrt(MU / r)
    return f_carrier * v * (RE / r) * np.cos(np.radians(elev_deg)) / C_LIGHT


def fspl_db(d, f):
    return 20 * np.log10(4 * np.pi * d * f / C_LIGHT)


def cn0_eff_db(c_dbw, n0_dbw_hz, ci_dbw, kappa):
    c = 10 ** (c_dbw / 10)
    n0 = 10 ** (n0_dbw_hz / 10)
    ci = 10 ** (ci_dbw / 10)
    return 10 * np.log10(c / (n0 + kappa * ci))


# --------------------------------------------------------------------------
# Scenario + plots
# --------------------------------------------------------------------------
def run(args):
    bw = args.bw
    victim = band_normalise(make_psd("bpsk", 1), bw)                 # GPS L1 C/A
    victim_name = "GPS L1 C/A  BPSK(1)"
    interferers = {
        "BPSK(1)": band_normalise(make_psd("bpsk", 1), bw),
        "BPSK(5)": band_normalise(make_psd("bpsk", 5), bw),
        "BOC(1,1)": band_normalise(make_psd("boc", 1, 1), bw),
        "BOC(6,1)": band_normalise(make_psd("boc", 1, 6), bw),
    }

    # ---- Fig 1: PSD validation (Welch vs analytic) ----
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (kind, n, m, lab) in zip(axs, [("bpsk", 1, None, "BPSK(1)"), ("boc", 1, 1, "BOC(1,1)")]):
        x, fs_samp = generate_signal(kind, n, m)
        fw, pw = welch(x, fs=fs_samp, nperseg=8192, return_onesided=False, scaling="density")
        fw, pw = np.fft.fftshift(fw), np.fft.fftshift(pw)
        fa = np.linspace(-6e6, 6e6, 4001)
        ga = make_psd(kind, n, m)(fa)
        ax.plot(fw / 1e6, 10 * np.log10(pw + 1e-30), lw=0.8, label="Welch estimate")
        ax.plot(fa / 1e6, 10 * np.log10(ga + 1e-30), "--", lw=1.2, label="analytic")
        ax.set_xlim(-6, 6); ax.set_ylim(-130, -55)
        ax.set_title(lab); ax.set_xlabel("frequency offset (MHz)"); ax.grid(alpha=0.3)
    axs[0].set_ylabel("PSD (dB/Hz)"); axs[0].legend()
    fig.tight_layout(); fig.savefig("figures/psd_validation.png", dpi=160); plt.close(fig)

    # ---- Fig 2: autocorrelation ----
    taus = np.linspace(-1.5 / F0, 1.5 / F0, 301)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for lab, p in [("BPSK(1)", interferers["BPSK(1)"]), ("BOC(1,1)", interferers["BOC(1,1)"]), ("BOC(6,1)", interferers["BOC(6,1)"])]:
        ax.plot(taus * F0, autocorrelation(p, bw, taus), label=lab)
    ax.set_xlabel("delay (chips)"); ax.set_ylabel("normalised ACF"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/acf.png", dpi=160); plt.close(fig)

    # ---- Fig 3: SSC vs frequency offset ----
    offs = np.linspace(-5e6, 5e6, 401)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for lab, p in interferers.items():
        ax.plot(offs / 1e6, [10 * np.log10(ssc(victim, p, bw, d, 12001)) for d in offs], label=lab)
    ax.set_xlabel("interferer carrier offset (MHz)"); ax.set_ylabel("SSC (dB/Hz)")
    ax.set_title(f"victim: {victim_name}"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("figures/ssc_vs_offset.png", dpi=160); plt.close(fig)

    # ---- Fig 4: C/N0 degradation vs elevation ----
    h, fcar = args.altitude * 1e3, args.freq
    elevs = np.linspace(5, 90, 86)
    d = slant_range(h, elevs)
    ci = args.eirp + args.rx_gain - fspl_db(d, fcar) - args.losses
    fig, ax = plt.subplots(figsize=(6, 3.8))
    rows = []
    for lab, p in interferers.items():
        deg = []
        for e, ci_e in zip(elevs, ci):
            dop = max_doppler(h, e, fcar)
            k = ssc(victim, p, bw, dop, 12001)
            deg.append((args.c_dbw - args.n0) - cn0_eff_db(args.c_dbw, args.n0, ci_e, k))
        deg = np.array(deg)
        ax.plot(elevs, deg, label=lab)
        rows.append((lab, deg.max(), elevs[deg.argmax()]))
    ax.axhline(args.threshold, color="k", ls=":", lw=1, label=f"threshold {args.threshold} dB")
    ax.set_xlabel("LEO elevation at user (deg)"); ax.set_ylabel("C/N0 degradation (dB)")
    ax.set_title(f"h = {args.altitude:.0f} km, EIRP = {args.eirp} dBW, f = {fcar/1e6:.2f} MHz")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("figures/cn0_degradation.png", dpi=160); plt.close(fig)

    print(f"Victim: {victim_name}, C/N0 nominal = {args.c_dbw - args.n0:.1f} dB-Hz, front-end BW = {bw/1e6:.1f} MHz")
    print(f"Geometry: h = {args.altitude:.0f} km | slant range {slant_range(h, 90)/1e3:.0f} km (zenith) "
          f"to {slant_range(h, 5)/1e3:.0f} km (5 deg) | max Doppler {max_doppler(h, 5, fcar)/1e3:.1f} kHz")
    print(f"{'Interferer':<10} {'worst degradation (dB)':>24} {'at elevation (deg)':>20}")
    for lab, dmax, e in rows:
        flag = "  <-- above threshold" if dmax > args.threshold else ""
        print(f"{lab:<10} {dmax:>24.3f} {e:>20.0f}{flag}")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bw", type=float, default=24e6, help="front-end bandwidth (Hz)")
    p.add_argument("--altitude", type=float, default=550.0, help="LEO altitude (km)")
    p.add_argument("--freq", type=float, default=1575.42e6, help="carrier frequency (Hz)")
    p.add_argument("--eirp", type=float, default=0.0, help="LEO-PNT EIRP towards user (dBW), illustrative")
    p.add_argument("--rx-gain", type=float, default=0.0, help="user antenna gain towards LEO (dBi)")
    p.add_argument("--losses", type=float, default=1.0, help="polarisation + atmospheric losses (dB)")
    p.add_argument("--c-dbw", type=float, default=-158.5, help="victim signal power at receiver (dBW)")
    p.add_argument("--n0", type=float, default=-201.5, help="noise density (dBW/Hz)")
    p.add_argument("--threshold", type=float, default=1.0, help="allowed C/N0 degradation (dB), assumption")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
