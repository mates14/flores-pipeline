"""
Small shared utilities used across the FLORES pipeline.

These reimplement (compactly, for a single-aperture instrument) algorithmic
ideas from ~/terka/pipeline/perek_pipelines/tools.py and orders.py, which were
written for the multi-order OES echelle pipeline. Kept local rather than
imported so this pipeline has no path dependency on that repo.
"""
import numpy as np
from astropy.stats import mad_std
from scipy.optimize import curve_fit
from scipy.special import erf


def gaussian(x, amplitude, mu, sigma):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_pixel_weights(y_pixels, y0, sigma):
    """
    Fraction of a Gaussian PSF centred at y0 (sigma) falling in each integer
    pixel of y_pixels. Weights sum to ~1 over a window wide enough to capture
    the PSF. Same convention as perek_pipelines/orders.py.
    """
    y_low = (y_pixels - 0.5 - y0) / (np.sqrt(2) * sigma)
    y_high = (y_pixels + 0.5 - y0) / (np.sqrt(2) * sigma)
    weights = 0.5 * (erf(y_high) - erf(y_low))
    total = np.sum(weights)
    if total > 0:
        weights = weights / total
    return weights


def fit_gaussian_centroid(x, y, mu_guess, sigma_guess, amp_guess=None,
                           sigma_bounds=(0.5, 20.0)):
    """
    Fit a single Gaussian to (x, y) and return (amplitude, mu, sigma, perr)
    or None if the fit fails / is unconstrained.
    """
    if len(x) < 4:
        return None
    if amp_guess is None:
        amp_guess = max(np.max(y) - np.median(y), 1e-6)
    lo = [0, mu_guess - sigma_guess * 3, sigma_bounds[0]]
    hi = [np.inf, mu_guess + sigma_guess * 3, sigma_bounds[1]]
    try:
        params, cov = curve_fit(
            gaussian, x, y, p0=[amp_guess, mu_guess, sigma_guess],
            bounds=(lo, hi), maxfev=2000,
        )
    except Exception:
        return None
    perr = np.sqrt(np.diag(cov))
    if not np.all(np.isfinite(perr)):
        return None
    return params[0], params[1], params[2], perr


def polyfit_reject(x, y, deg=3, thres=(10, 5, 3, 2), thres_max=None):
    """
    Robust polynomial fit with iterative sigma-clipping on the residuals
    (MAD-based), same rejection strategy as
    perek_pipelines/tools.py:polyfit_reject / curve_fit_reject.

    Returns (coeffs, mask_good) where mask_good is relative to the input
    (unsorted) x, y arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.ones(len(x), dtype=bool)
    coeffs = np.polyfit(x, y, deg)

    for t in thres:
        if np.sum(mask) <= deg + 1:
            break
        coeffs = np.polyfit(x[mask], y[mask], deg)
        resid = np.abs(np.polyval(coeffs, x) - y)
        std = mad_std(resid[mask])
        if std == 0:
            # perfect fit (zero scatter) - nothing to reject, and cut=0
            # would otherwise flag every point via the strict "<" below
            continue
        if thres_max is not None:
            cut = max(std * t, thres_max)
        else:
            cut = std * t
        mask = resid < cut

    return coeffs, mask


def fill_nan(y):
    """Replace NaNs in a 1D array by linear interpolation."""
    y = np.array(y, dtype=float)
    nans = np.isnan(y)
    if np.all(nans) or not np.any(nans):
        return y
    idx = np.arange(len(y))
    y[nans] = np.interp(idx[nans], idx[~nans], y[~nans])
    return y
