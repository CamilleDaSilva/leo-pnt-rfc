# LEO-PNT RF compatibility simulator

Python toolkit to study how a LEO-PNT signal-in-space affects existing GNSS/RNSS receivers
(GPS L1 C/A, Galileo E1).

## What it does
- Generates BPSK-R(n) and sine-phased BOC(m,n) signals; checks the analytic PSD against a Welch estimate
- Autocorrelation function (acquisition / tracking shape)
- Spectral Separation Coefficient (SSC) between interferer and victim vs carrier offset (Doppler included)
- Link budget from orbital geometry (altitude, elevation, slant range, FSPL, Doppler) and effective C/N0 degradation

## Run
```bash
pip install numpy scipy matplotlib
python leo_pnt_rfc.py                       # default scenario
python leo_pnt_rfc.py --eirp 10 --altitude 600 --threshold 0.5
```
Figures are written to `figures/`.

## Method
Effective C/N0 (Betz): `(C/N0)_eff = C / (N0 + kappa * C_I)`, with `kappa` the SSC (Hz^-1),
PSDs normalised to unit power in the receiver front-end bandwidth.

## Limits (to state honestly)
- EIRP, noise density, victim power and the degradation threshold are illustrative parameters, not mission data
- Single interferer, circular orbit in the user's plane, Earth rotation ignored, no antenna patterns yet
- Closed-form BOC PSD only for 2m/n even; CBOC (Galileo E1 OS) is approximated by BOC(1,1)

## Ideas to extend
Antenna patterns, constellation aggregate interference, other bands (L5/E5, S-band), correlator-level receiver simulation.
