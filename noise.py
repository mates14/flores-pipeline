"""
Per-pixel flux uncertainty for an extracted FLORES spectrum.

Primary method: the standard CCD equation applied to the Gaussian-weighted
aperture sum that trace.py produces (see trace.py:extract_aperture
docstring for the weighting convention). Needs GAIN/RDNOISE header keywords.

Fallback when those aren't available/characterized: a local-scatter
estimator in the spirit of perek_pipelines/estimate_noise.py (itself based
on the DER_SNR method, https://stdatu.stsci.edu/vodocs/der_snr.pdf) -
generic 1D-spectrum code, reimplemented compactly here rather than imported
since it has no echelle-specific dependency either way.
"""
import numpy as np


def ccd_equation_error(extraction, gain, read_noise_e):
    """
    extraction: dict as returned by trace.extract_aperture.
    gain: e-/ADU. read_noise_e: read noise in electrons (per pixel).

    var(flux) = (weighted_raw * gain + sumw2 * read_noise_e**2) / gain**2
    i.e. the CCD equation (shot noise on the raw, pre-background-subtraction
    counts, plus read noise) propagated through the same P_i-weighted sum
    used to build `flux`, under the standard assumption of ~uniform variance
    across the few-pixel aperture window.
    """
    weighted_raw = extraction["weighted_raw"]
    sumw2 = extraction["sumw2"]
    var = (weighted_raw * gain + sumw2 * read_noise_e ** 2) / gain ** 2
    var[var <= 0] = np.nan
    return np.sqrt(var)


def local_scatter_error(flux, order=3):
    """
    DER_SNR-style local noise estimate from the flux array itself, for use
    when GAIN/RDNOISE aren't known. Window size follows
    perek_pipelines/estimate_noise.py's heuristic (~1/150 of the spectrum
    length, floored/ceiled to [20, len/10] pixels).
    """
    n = len(flux)
    win = int(np.clip(n / 150 * 5, 20, max(20, n // 10)))
    err = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win)
        seg = flux[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 6:
            continue
        # 3rd-order median-of-differences noise estimator (DER_SNR)
        s = seg
        m = len(s)
        f3 = 0.6052697319
        err[i] = f3 * np.median(np.abs(2.0 * s[2:m - 2] - s[0:m - 4] - s[4:m])) \
            if m > 4 else np.nanstd(seg)

    nans = ~np.isfinite(err)
    if np.any(nans) and not np.all(nans):
        idx = np.arange(n)
        err[nans] = np.interp(idx[nans], idx[~nans], err[~nans])
    return err


def estimate_error(extraction, header=None):
    """
    Pick CCD-equation propagation when GAIN/RDNOISE headers are present,
    otherwise fall back to the local-scatter estimator on the flux array.
    Returns (error_array, method_str).
    """
    gain = None
    read_noise = None
    if header is not None:
        for key in ("GAIN", "EGAIN"):
            if key in header:
                gain = float(header[key])
                break
        for key in ("RDNOISE", "RON", "READNOIS"):
            if key in header:
                read_noise = float(header[key])
                break

    if gain and read_noise:
        return ccd_equation_error(extraction, gain, read_noise), "ccd_equation"

    return local_scatter_error(extraction["flux"]), "local_scatter"
