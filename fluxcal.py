"""
Flux calibration from spectrophotometric standards.

Everything reduce_flores.py produces is *relative* flux (ADU, no
flat-fielding or response correction - see its module docstring). This
module derives an instrumental response curve from standard stars with
known absolute flux and applies it to convert any reduced spectrum to
physical units (erg/s/cm2/A).

No FITS header in this dataset carries airmass/altitude, so there is no
way to apply a proper per-frame atmospheric extinction correction - the
derived response bakes in whatever airmass mix the standards were observed
at, and applying it to a target observed at a different airmass carries an
uncorrected systematic error (worse toward the blue). Using several
standards spread across the night partially averages this out but does not
fix it. Every calibrated output's header says so explicitly rather than
implying a precision the data doesn't support.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.stats import mad_std
from scipy.interpolate import PchipInterpolator

SPEED_OF_LIGHT_ANGSTROM_S = 2.99792458e18  # for the Fnu(Jy) -> Flambda relation below

# Real stellar absorption lines (Balmer series - all four standards are
# early type, A0V/B-type) plus telluric bands, masked out before fitting
# the response: telluric depth isn't rescalable between different
# airmasses without extinction data, so it must not be baked into the
# response curve, or every target would get the *standard's* telluric
# absorption partially removed instead of its own. Same band list this
# codebase has used since the very first coadd/response-shaped work
# (perek_pipelines/echelle_reduction.py:poly_normalization).
IGNORE_WINDOWS = [
    (3831.4, 3839.4), (3883, 3893), (3933 - 3, 3933 + 3),
    (3963.5, 3981), (4090, 4115), (4320, 4355),
    (4842, 4888), (6540, 6590), (6860, 6880),
    (6888.1, 6890.5), (6892, 6893.6), (7590, 7617), (7622.8, 7625),
]


def _fnu_jy_to_flambda(fnu_jy, wl_angstrom):
    """Fnu[Jy] -> Flambda[erg/s/cm2/A]. Standard relation Flambda = Fnu*c/lambda**2,
    with the unit-conversion constants folded in (verified this session
    against Vega's well-known V-band flux, ~3.5e-9 erg/s/cm2/A at ~3540 Jy,
    5500A)."""
    return np.asarray(fnu_jy) * 2.99792458e-5 / np.asarray(wl_angstrom) ** 2


def load_calspec(path):
    """HST CALSPEC bintable: WAVELENGTH[A] / FLUX[FLAM=erg/s/cm2/A] directly."""
    with fits.open(path) as hdul:
        data = hdul[1].data
        wl = np.asarray(data["WAVELENGTH"], dtype=float)
        flux = np.asarray(data["FLUX"], dtype=float)
    order = np.argsort(wl)
    return wl[order], flux[order]


def load_maestro_tab(path):
    """
    MAESTRO/Turnshek-derived .tab format: header comment lines starting
    with '*', a 'SET .Z.UNITS = "milli-Janskys"' line, then two columns
    (wavelength[A], flux).

    The header's "milli-Janskys" label is wrong (verified against the
    local CALSPEC Vega spectrum this session: treating the raw numbers as
    mJy gives a V-band flux exactly 1000x the true CALSPEC value; treating
    them as micro-Jy matches CALSPEC to ~0.3%). Trusting the measurement,
    not the 2002-vintage label.
    """
    wl, flux_ujy = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("*") or s.upper().startswith("SET "):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            try:
                w, v = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            wl.append(w)
            flux_ujy.append(v)
    wl = np.array(wl)
    fnu_jy = np.array(flux_ujy) / 1e6
    order = np.argsort(wl)
    wl = wl[order]
    return wl, _fnu_jy_to_flambda(fnu_jy[order], wl)


def load_iraf_onedstd(path):
    """
    IRAF onedstds format: wavelength[A], AB magnitude, bandpass[A]
    (bandpass unused here). AB = -2.5*log10(Fnu) - 48.60, Fnu in
    erg/s/cm2/Hz.
    """
    wl, mag = [], []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            wl.append(float(parts[0]))
            mag.append(float(parts[1]))
    wl = np.array(wl)
    fnu_cgs = 10 ** (-0.4 * (np.array(mag) + 48.60))  # erg/s/cm2/Hz
    fnu_jy = fnu_cgs / 1e-23
    order = np.argsort(wl)
    wl = wl[order]
    return wl, _fnu_jy_to_flambda(fnu_jy[order], wl)


def _mask_ignore_windows(wl):
    mask = np.ones(len(wl), dtype=bool)
    for lo, hi in IGNORE_WINDOWS:
        mask &= ~((wl > lo) & (wl < hi))
    return mask


class ResponseCurve:
    """count-rate(lambda) / Flambda_reference(lambda), as a function of
    wavelength. Only valid within [wl_min, wl_max] - no extrapolation."""

    def __init__(self, wl, response):
        order = np.argsort(wl)
        self.wl = np.asarray(wl)[order]
        self.response = np.asarray(response)[order]
        self.wl_min = float(self.wl[0])
        self.wl_max = float(self.wl[-1])
        self._interp = PchipInterpolator(self.wl, self.response, extrapolate=False)

    def __call__(self, wl):
        return self._interp(wl)


def derive_response_one(reduced_wl, reduced_flux, exptime_s, ref_wl, ref_flambda,
                        min_points=20):
    """
    One standard's response curve: count-rate(lambda) / Flambda_ref(lambda),
    masked at known stellar lines/telluric bands and smoothed with a
    monotone-safe PCHIP fit through the surviving points (no assumed
    functional form - same approach validated for the dispersion solution).

    reduced_wl/reduced_flux: this standard's already-reduced spectrum
    (from reduce_flores.py's output). exptime_s: the source frame's
    EXPTIME, to convert flux (total ADU) to a count *rate*.

    Returns a ResponseCurve, or raises RuntimeError if too few usable
    points remain after masking/range-clipping.
    """
    count_rate = reduced_flux / exptime_s

    lo = max(reduced_wl.min(), ref_wl.min())
    hi = min(reduced_wl.max(), ref_wl.max())
    in_range = (reduced_wl >= lo) & (reduced_wl <= hi)
    wl = reduced_wl[in_range]
    rate = count_rate[in_range]

    ref_interp = PchipInterpolator(ref_wl, ref_flambda, extrapolate=False)
    ref_on_obs = ref_interp(wl)

    good = np.isfinite(wl) & np.isfinite(rate) & np.isfinite(ref_on_obs)
    good &= (ref_on_obs > 0) & _mask_ignore_windows(wl)
    if np.sum(good) < min_points:
        raise RuntimeError(
            f"only {np.sum(good)} usable points to derive a response curve "
            f"(need >= {min_points})"
        )

    wl_good = wl[good]
    ratio = rate[good] / ref_on_obs[good]

    # bin to a coarser, evenly spaced grid before the PCHIP fit: reduces
    # per-pixel noise and duplicate/near-duplicate wavelengths (PCHIP
    # needs strictly increasing x), same spirit as trace.py's bin-then-fit
    # pattern for the trace position.
    n_bins = max(min_points, min(200, len(wl_good) // 5))
    edges = np.linspace(wl_good.min(), wl_good.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(wl_good, edges) - 1, 0, n_bins - 1)
    bin_wl, bin_ratio = [], []
    for i in range(n_bins):
        sel = bin_idx == i
        if np.sum(sel) < 2:
            continue
        vals = ratio[sel]
        med = np.median(vals)
        robust = np.abs(vals - med) < 5 * max(mad_std(vals), med * 1e-3)
        if np.sum(robust) == 0:
            robust = np.ones(len(vals), dtype=bool)
        bin_wl.append(np.median(wl_good[sel][robust]))
        bin_ratio.append(np.median(vals[robust]))

    if len(bin_wl) < 4:
        raise RuntimeError(f"only {len(bin_wl)} bins survived - too little coverage")

    return ResponseCurve(np.array(bin_wl), np.array(bin_ratio))


def combine_responses(responses, grid_points=400):
    """
    Robust combination of several standards' ResponseCurves, in log space
    (response is a multiplicative/throughput quantity). Combines only
    where at least one standard has coverage - no extrapolation beyond
    the union of the individual curves' valid ranges.
    """
    if not responses:
        raise RuntimeError("no response curves to combine")

    wl_min = min(r.wl_min for r in responses)
    wl_max = max(r.wl_max for r in responses)
    grid = np.linspace(wl_min, wl_max, grid_points)

    combined_wl, combined_resp = [], []
    for w in grid:
        vals = []
        for r in responses:
            if r.wl_min <= w <= r.wl_max:
                v = float(r(w))
                if np.isfinite(v) and v > 0:
                    vals.append(np.log(v))
        if not vals:
            continue
        vals = np.array(vals)
        if len(vals) >= 3:
            std = mad_std(vals)
            if std > 0:
                keep = np.abs(vals - np.median(vals)) < 5 * std
                if np.any(keep):
                    vals = vals[keep]
        combined_wl.append(w)
        combined_resp.append(np.exp(np.median(vals)))

    return ResponseCurve(np.array(combined_wl), np.array(combined_resp))
