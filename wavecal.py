"""
Bootstrapped wavelength calibration for FLORES.

Starts from a one-time seed line list (seeds/flores_seed_linelist.csv,
converted once from a manual IRAF `identify` run - see seeds/convert_seed.py)
and, for every new comparison-lamp extraction, re-locates those lines and
iteratively grows the line table against a full ThAr atlas
(thar_lovis_pepe_clean.csv, carried over from perek_pipelines).

Unlike perek_pipelines/calibrate.py (built for echelle orders), the dispersion
model here is NOT a single global polynomial: FLORES's resolution runs from
~2000 in the blue to ~200 in the NIR, which makes a global polynomial fit
non-monotonic and prone to bad extrapolation into low-throughput dead zones
(see project memory on this instrument). Instead we use a monotonic cubic
Hermite interpolant (PCHIP) through the accepted (pixel, wavelength) points -
the scipy equivalent of IRAF's "spline3, few pieces" fix for this instrument,
with the added guarantee of no oscillation between points and no
extrapolation past the outermost identified line.
"""
import csv
import numpy as np
from scipy.interpolate import PchipInterpolator

from astropy.stats import mad_std

from common import fit_gaussian_centroid


class DispersionSolution:
    def __init__(self, pixel, wavelength, fwhm_pix, rms_pix, n_lines, coarse_shift_px=0):
        self.coarse_shift_px = coarse_shift_px
        order = np.argsort(pixel)
        self.pixel = np.asarray(pixel)[order]
        self.wavelength = np.asarray(wavelength)[order]
        self.fwhm_pix = np.asarray(fwhm_pix)[order]
        self.rms_pix = rms_pix
        self.n_lines = n_lines
        self.px_min = self.pixel[0]
        self.px_max = self.pixel[-1]
        self._fwd = PchipInterpolator(self.pixel, self.wavelength, extrapolate=False)
        # inverse: wavelength -> pixel. Forward is monotonic by construction
        # (deduplicated, sorted pixels with a strictly increasing or
        # strictly decreasing wavelength), so this is just the same points
        # resorted by wavelength.
        wl_order = np.argsort(self.wavelength)
        self._inv = PchipInterpolator(
            self.wavelength[wl_order], self.pixel[wl_order], extrapolate=False
        )

    def pixel_to_wavelength(self, pixels):
        """Wavelength at given pixel(s); NaN outside the calibrated range."""
        return self._fwd(pixels)

    def wavelength_to_pixel(self, wavelengths):
        return self._inv(wavelengths)


def load_seed_linelist(path):
    pixel, wavelength, fwhm = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            pixel.append(float(row["pixel"]))
            wavelength.append(float(row["wavelength_air"]))
            fwhm.append(float(row["fwhm_guess_pix"]))
    return np.array(pixel), np.array(wavelength), np.array(fwhm)


def load_thar_atlas(path, min_intensity=None):
    """
    Load the ThAr atlas CSV (perek_pipelines/thar_lovis_pepe_clean.csv format:
    wave_air, int, ID, ... columns). Returns (wavelengths, intensities).
    """
    wl, intensity = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                w = float(row["wave_air"])
                i = float(row["int"])
            except (KeyError, ValueError):
                continue
            wl.append(w)
            intensity.append(i)
    wl = np.array(wl)
    intensity = np.array(intensity)
    if min_intensity is not None:
        mask = intensity >= min_intensity
        wl, intensity = wl[mask], intensity[mask]
    order = np.argsort(wl)
    return wl[order], intensity[order]


def _local_outlier_mask(pixel, wavelength, window=21, nsigma=4, min_neighbors=4):
    """
    Robust outlier rejection using a *local* leave-one-out quadratic fit
    (nearest `window` points in pixel order) instead of one global
    polynomial. FLORES's dispersion isn't well described by any single
    low-order global polynomial (resolution runs from ~2000 in the blue to
    ~200 in the NIR - see module docstring), so a global reference fit
    over-penalises whichever region has the least curvature. Fitting locally
    adapts to that instead of fighting it.

    pixel, wavelength must already be sorted by pixel.
    """
    n = len(pixel)
    resid = np.zeros(n)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = list(range(lo, i)) + list(range(i + 1, hi))
        if len(idx) < min_neighbors:
            continue
        coeffs = np.polyfit(pixel[idx], wavelength[idx], 2)
        resid[i] = wavelength[i] - np.polyval(coeffs, pixel[i])
    std = mad_std(resid)
    if std == 0 or not np.isfinite(std):
        return np.ones(n, dtype=bool), resid, std
    return np.abs(resid) < nsigma * std, resid, std


def _fit_lines_near(spectrum, pixel_guess, window=8, sigma_guess=1.8,
                     sigma_bounds=(0.5, 9.5)):
    """
    Gaussian-fit the spectrum around each guessed pixel position.
    Returns arrays (position, sigma, fwhm) with NaN for failed fits.

    The fit window for each line is capped well below its distance to the
    nearest other guessed line, so a tight blend can't have one line's fit
    wander onto its neighbour (the seed list has lines as close as ~2 px
    apart in the crowded blue end - a fixed 8px window would straddle
    several of them).
    """
    npix = len(spectrum)
    x_all = np.arange(npix) + 1
    positions = np.full(len(pixel_guess), np.nan)
    fwhm = np.full(len(pixel_guess), np.nan)

    pixel_guess = np.asarray(pixel_guess, dtype=float)

    for i, guess in enumerate(pixel_guess):
        if len(pixel_guess) > 1:
            other = np.delete(pixel_guess, i)
            local_gap = np.min(np.abs(other - guess))
        else:
            local_gap = window * 3
        local_window = min(window, max(2.5, 0.45 * local_gap))

        lo = int(np.floor(guess - local_window))
        hi = int(np.ceil(guess + local_window))
        if lo < 0 or hi >= npix:
            continue
        xwin = x_all[lo:hi]
        ywin = spectrum[lo:hi]
        fit = fit_gaussian_centroid(xwin, ywin, guess, sigma_guess,
                                     sigma_bounds=sigma_bounds)
        if fit is None:
            continue
        _, mu, sigma, _ = fit
        positions[i] = mu
        fwhm[i] = sigma * 2.3548200450309493  # 2*sqrt(2*ln2)

    return positions, fwhm


def _coarse_pixel_shift(spectrum, seed_pixel, max_shift=60, comb_sigma=2.0):
    """
    Estimate an overall pixel registration shift between `spectrum` and the
    seed line positions, by cross-correlating the spectrum against a
    synthetic comb of narrow peaks placed at the seed pixels.

    The seed was calibrated on one specific night; a different night's
    comp frame can land systematically offset from it by several to a few
    tens of pixels (thermal/mechanical drift since the seed was taken) even
    though the *shape* of the dispersion relation is unchanged. Without
    this, the small per-line search windows in _fit_lines_near - deliberately
    tight, to avoid jumping to a neighbouring line in the crowded blue end -
    can miss every real line and lock onto whatever's coincidentally nearby
    instead, which produces a fit that looks like it has plenty of lines but
    is actually wrong (see wavecal module history / project notes on this).

    Returns the integer pixel shift to ADD to seed_pixel to align it with
    `spectrum` (0 if the arrays are too short to search).
    """
    npix = len(spectrum)
    if npix < 4 * max_shift:
        return 0
    x = np.arange(npix) + 1.0
    comb = np.zeros(npix)
    for p in seed_pixel:
        comb += np.exp(-0.5 * ((x - p) / comb_sigma) ** 2)
    obs = np.nan_to_num(np.clip(spectrum - np.nanmedian(spectrum), 0, None))
    obs_max = np.max(obs)
    if obs_max <= 0:
        return 0
    obs = obs / obs_max
    comb = comb / np.max(comb)

    corr = np.correlate(obs, comb, mode="full")
    lags = np.arange(-(npix - 1), npix)
    mask = np.abs(lags) <= max_shift
    return int(lags[mask][np.argmax(corr[mask])])


def solve_dispersion(comparison_spectrum, seed_pixel, seed_wavelength,
                      thar_wl=None, thar_intensity=None,
                      window=8, max_iterations=1,
                      min_lines=8, max_pixel_rms=1.3,
                      fwhm_bounds=(1.2, 25.0),
                      atlas_min_separation_pix=3.0,
                      atlas_min_intensity_percentile=75.0,
                      reject_window=21, reject_nsigma=4,
                      coarse_register=True, coarse_max_shift=60):
    """
    Bootstrap a DispersionSolution for one comparison-lamp extraction,
    starting from the seed line list and, optionally, iteratively growing it
    against the ThAr atlas.

    Default is max_iterations=1 (seed lines only, no atlas growth). Atlas
    growth (max_iterations=2+) was validated against IRAF's own carefully
    tuned extraction (260716/*.ms.txt) and clearly helped there (122 -> 178
    lines, rms 0.70 -> 1.02 px). But against this pipeline's own
    trace.py extraction - rougher, more blended lines, no dedicated
    background frame - it just as clearly hurt (128 -> 180 lines, rms 0.89
    -> 11 px on the same test frame): a few mismatched blended lines early
    on corrupt the pixel->wavelength prediction used to place later
    candidates, and errors compound. Re-enable growth once trace.py's
    extraction quality has been validated on real (not commissioning-era)
    frames.

    coarse_register (default True) cross-correlates the spectrum against a
    synthetic comb of the seed positions first and shifts the seed to match
    (see _coarse_pixel_shift) - needed whenever this comp wasn't taken on
    the same night/session the seed was derived from; harmless (shift ~= 0)
    when it was.

    Raises RuntimeError (with a clear reason) if the fit doesn't meet the
    quality gates - callers should skip the paired science frame rather than
    emit a bad calibration.
    """
    if thar_wl is not None and thar_intensity is not None and len(thar_wl):
        thresh = np.percentile(thar_intensity, atlas_min_intensity_percentile)
        strong = thar_intensity >= thresh
        thar_wl = thar_wl[strong]

    seed_pixel = np.array(seed_pixel, dtype=float)
    coarse_shift = 0
    if coarse_register:
        coarse_shift = _coarse_pixel_shift(
            comparison_spectrum, seed_pixel, max_shift=coarse_max_shift
        )
        seed_pixel = seed_pixel + coarse_shift

    linetable_px = seed_pixel
    linetable_wl = np.array(seed_wavelength, dtype=float)

    result = None
    for iteration in range(max_iterations):
        pos, fwhm = _fit_lines_near(comparison_spectrum, linetable_px, window=window)
        good = np.isfinite(pos) & np.isfinite(fwhm)
        good &= (fwhm >= fwhm_bounds[0]) & (fwhm <= fwhm_bounds[1])

        if np.sum(good) < min_lines:
            raise RuntimeError(
                f"only {np.sum(good)} usable comparison lines "
                f"(need >= {min_lines}) at iteration {iteration}"
            )

        px = pos[good]
        wl = linetable_wl[good]
        fw = fwhm[good]

        order = np.argsort(px)
        px, wl, fw = px[order], wl[order], fw[order]

        # de-duplicate / enforce minimum pixel spacing so PCHIP gets a
        # strictly increasing x array
        keep = np.ones(len(px), dtype=bool)
        for i in range(1, len(px)):
            if px[i] - px[i - 1] < 0.5:
                keep[i] = False
        px, wl, fw = px[keep], wl[keep], fw[keep]

        # robust outlier rejection, local rather than global (see
        # _local_outlier_mask docstring for why)
        mask_good, resid, _ = _local_outlier_mask(
            px, wl, window=reject_window, nsigma=reject_nsigma
        )
        if np.sum(mask_good) < min_lines:
            raise RuntimeError(
                f"only {np.sum(mask_good)} lines survive outlier rejection "
                f"(need >= {min_lines}) at iteration {iteration}"
            )

        # pixel-normalised rms of the surviving lines' local-fit residuals,
        # for the quality gate (Angstrom alone is meaningless: Angstrom/pixel
        # varies ~10x across the band)
        local_disp = np.gradient(wl, px)
        local_disp[local_disp == 0] = np.nanmedian(np.abs(local_disp[local_disp != 0]))
        rms_pix = float(np.sqrt(np.mean(
            (resid[mask_good] / local_disp[mask_good]) ** 2
        )))

        px, wl, fw = px[mask_good], wl[mask_good], fw[mask_good]
        result = DispersionSolution(px, wl, fw, rms_pix, len(px),
                                     coarse_shift_px=coarse_shift)

        last_iteration = (iteration == max_iterations - 1)
        if thar_wl is None or last_iteration:
            break

        # predict where more atlas lines should fall given the current
        # solution, keep the ones well clear of already-matched lines
        wmin = float(np.min(wl)) + 1.0
        wmax = float(np.max(wl)) - 1.0
        atlas_mask = (thar_wl > wmin) & (thar_wl < wmax)
        if not np.any(atlas_mask):
            break
        cand_wl = thar_wl[atlas_mask]
        cand_px = result.wavelength_to_pixel(cand_wl)
        valid = np.isfinite(cand_px)
        cand_wl, cand_px = cand_wl[valid], cand_px[valid]

        unmatched = np.ones(len(cand_px), dtype=bool)
        for i, p in enumerate(cand_px):
            if np.any(np.abs(px - p) < atlas_min_separation_pix):
                unmatched[i] = False
        cand_wl, cand_px = cand_wl[unmatched], cand_px[unmatched]
        if len(cand_px) == 0:
            break

        linetable_px = np.concatenate([px, cand_px])
        linetable_wl = np.concatenate([wl, cand_wl])

    if result.rms_pix > max_pixel_rms:
        raise RuntimeError(
            f"dispersion fit rms {result.rms_pix:.2f} px exceeds "
            f"threshold {max_pixel_rms:.2f} px ({result.n_lines} lines)"
        )

    return result
