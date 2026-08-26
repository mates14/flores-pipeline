"""
Multi-aperture trace finding, tilt correction, and optimal extraction for
FLORES.

FLORES feeds two fibers (two separate sky positions) into every raw frame,
~18 rows apart - confirmed on real data, see project history. `find_traces`
locates both (or however many are actually visible), not just the
brightest, and `extract_aperture` keeps each fiber's aperture/background
bands from reaching into its neighbour's light.

One of the two fibers (whichever the user chooses for that observation) can
show a real geometric tilt: its spectral lines aren't perpendicular to the
dispersion axis, so straight column-summing blurs it. `measure_tilt` +
`TiltModel` measure and correct that directly from a comp frame (a
session-level calibration, like the wavelength solution - not derived from
the science trace itself, which is poorly conditioned for this).

Extraction is two-pass "optimal extraction" (Horne 1986): a quick first
pass gives a rough spectrum, which is used to build an empirical spatial
profile (no assumed functional form - the true profile is a boxcar
(fibre aperture) convolved with the optical PSF, not a clean Gaussian),
allowed to vary smoothly along the dispersion axis. The second pass
re-extracts with that profile, which also folds in cosmic-ray rejection
(a pixel that doesn't fit the profile-scaled prediction is flagged) instead
of a separate ad hoc step.
"""
import numpy as np
from astropy.stats import mad_std
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from common import fit_gaussian_centroid, gaussian_pixel_weights, polyfit_reject


class TraceModel:
    def __init__(self, coeffs, sigma, x_min, x_max, tag=None):
        self.coeffs = coeffs
        self.sigma = sigma
        self.x_min = x_min
        self.x_max = x_max
        self.tag = tag  # set by measure_tilt: "tilted" / "untilted"

    def y_of_x(self, x):
        return np.polyval(self.coeffs, x)


def find_traces(frame2d, bin_width=20, min_bin_flux_sigma=5.0,
                trace_poly_deg=3, sigma_guess=2.0, sigma_bounds=(0.7, 8.0),
                reject_thres=(8, 5, 3), peak_min_distance=10, max_traces=4,
                peak_smooth_sigma=3.0):
    """
    Locate every spectral trace (fibre) in a 2D (ny, nx) frame.

    Returns a list of TraceModel, sorted by row position. Raises
    RuntimeError if nothing usable is found at all.
    """
    ny, nx = frame2d.shape
    y = np.arange(ny)

    # per dispersion-bin: find every peak above threshold, not just the
    # brightest, and fit each one
    bin_peaks = []  # list of (x_center, [(mu, sigma), ...])
    for x0 in range(0, nx, bin_width):
        x1 = min(nx, x0 + bin_width)
        profile = np.median(frame2d[:, x0:x1], axis=1)
        baseline = np.median(profile)
        amplitude = profile - baseline
        noise = np.std(amplitude[amplitude < np.percentile(amplitude, 60)])
        if noise <= 0:
            continue
        # peak-find on a smoothed copy: a single fibre's profile is a
        # flat-topped boxcar-convolved-with-PSF shape (not a clean
        # Gaussian - see module docstring), and under noise a flat top
        # readily throws up multiple spurious local maxima a few pixels
        # apart. Smoothing merges those while still keeping the two real
        # fibres (~18px apart) well separated; the actual centroid fit
        # below still uses the unsmoothed data.
        smoothed = gaussian_filter1d(amplitude, sigma=peak_smooth_sigma)
        peak_idxs, _ = find_peaks(smoothed, height=min_bin_flux_sigma * noise,
                                  distance=peak_min_distance,
                                  prominence=min_bin_flux_sigma * noise * 0.6)
        found = []
        for peak_idx in peak_idxs:
            fit = fit_gaussian_centroid(
                y.astype(float), amplitude, float(peak_idx), sigma_guess,
                sigma_bounds=sigma_bounds,
            )
            if fit is None:
                continue
            _, mu, sigma, _ = fit
            if mu < 0 or mu > ny - 1:
                continue
            found.append((mu, sigma))
        if found:
            bin_peaks.append(((x0 + x1) / 2, found))

    if not bin_peaks:
        raise RuntimeError("no trace peaks found in any bin - frame may have no visible trace")

    # associate each bin's peaks into per-fibre sequences by nearest-row
    # matching to that sequence's most recent point (traces don't cross)
    sequences = []  # each: {"x": [...], "y0": [...], "sigma": [...]}
    for x_center, peaks in bin_peaks:
        used = set()
        for mu, sigma in peaks:
            best_i, best_dist = None, None
            for i, seq in enumerate(sequences):
                if i in used or not seq["x"]:
                    continue
                dist = abs(mu - seq["y0"][-1])
                if dist < peak_min_distance * 1.5 and (best_dist is None or dist < best_dist):
                    best_i, best_dist = i, dist
            if best_i is None:
                sequences.append({"x": [x_center], "y0": [mu], "sigma": [sigma]})
                used.add(len(sequences) - 1)
            else:
                sequences[best_i]["x"].append(x_center)
                sequences[best_i]["y0"].append(mu)
                sequences[best_i]["sigma"].append(sigma)
                used.add(best_i)

    traces = []
    for seq in sequences:
        xs, y0s, sigmas = np.array(seq["x"]), np.array(seq["y0"]), np.array(seq["sigma"])
        if len(xs) < trace_poly_deg + 2:
            continue
        coeffs, mask_good = polyfit_reject(xs, y0s, deg=trace_poly_deg, thres=reject_thres)
        if np.sum(mask_good) < trace_poly_deg + 2:
            continue
        sigma = float(np.median(sigmas[mask_good]))
        traces.append(TraceModel(coeffs, sigma, x_min=float(xs.min()), x_max=float(xs.max())))

    if not traces:
        raise RuntimeError("no trace survived sequence-building/outlier rejection")

    traces.sort(key=lambda t: t.y_of_x((t.x_min + t.x_max) / 2))
    return traces[:max_traces]


class TiltModel:
    """shift(dy): pixel shift along the dispersion axis for a row `dy`
    pixels above/below the trace centre. Linear in dy; ~0 for an untilted
    fibre."""

    def __init__(self, slope, rms_px):
        self.slope = slope
        self.rms_px = rms_px

    def shift_of_dy(self, dy):
        return self.slope * dy

    def is_tilted(self, threshold_px_per_row=0.07):
        """
        0.07 sits in the clean gap between the two real populations seen
        across many comps this session: the genuinely tilted fibre
        consistently measures ~0.11-0.15 px/row, the untilted one ~0.01-0.05
        (not exactly 0 - real measurement noise floor). The previous 0.03
        threshold sat inside that noise floor: on at least one comp the
        untilted fibre's slope (-0.0487) crossed it and got misclassified
        as tilted, which made extract_aperture wrongly de-shear a fibre
        that didn't need it - corrupting its wavelength registration badly
        enough (500+ px rms) to fail the wavecal quality gate and vanish
        from the comp's fibre dict entirely, silently leaving only the
        genuinely-tilted fibre's solution for anything to match against.
        """
        return abs(self.slope) > threshold_px_per_row


def measure_tilt(frame2d, trace, n_slices=7, max_shift=8, min_slice_flux_sigma=8.0):
    """
    Measure a fibre's geometric tilt from a comparison (ThAr) frame: slice
    the aperture into thin single-row slices, cross-correlate each against
    the centre row, fit a line through (row offset, pixel shift).

    A comp frame gives many bright, narrow, well-separated lines - the
    well-conditioned signal a tilt measurement needs (unlike the science
    trace's continuum). Session-level calibration: measure once per comp
    per fibre, reuse for every science frame from that fibre.

    The true shift-per-row is typically well under 1px (FLORES's measured
    tilt is of order 0.1px/row), so a plain integer-lag np.correlate peak
    is nowhere near good enough - at a handful of rows' offset the true
    shift is still sub-pixel and just rounds to 0 for most rows, and the
    fitted slope ends up dominated by whichever single row happened to
    round the "wrong" way (verified: this produced a fitted slope whose
    entire signal was one outlier row, and on a different comp frame with
    slightly different noise, the same instrument tilt measured as exactly
    zero for both fibres - a real bug, not just noise). Each row's peak is
    refined to sub-pixel precision by parabolic interpolation of the
    correlation values around the integer peak (standard 3-point
    quadratic peak refinement).

    Returns a TiltModel. Raises RuntimeError if there isn't enough signal
    to measure it (too few usable row slices).
    """
    ny, nx = frame2d.shape
    x_center = (trace.x_min + trace.x_max) / 2
    y0 = int(round(trace.y_of_x(x_center)))
    half = n_slices // 2
    offsets = np.arange(-half, half + 1)

    ref_row = frame2d[y0, :]
    ref = np.nan_to_num(np.clip(ref_row - np.median(ref_row), 0, None))
    ref_level = np.std(ref)

    offs_used, shifts = [], []
    for off in offsets:
        row_idx = y0 + off
        if row_idx < 0 or row_idx >= ny:
            continue
        row = frame2d[row_idx, :]
        obs = np.nan_to_num(np.clip(row - np.median(row), 0, None))
        if np.std(obs) < min_slice_flux_sigma * 1.0 or ref_level == 0:
            continue
        corr = np.correlate(obs, ref, mode="full")
        lags = np.arange(-(nx - 1), nx)
        mask = np.abs(lags) <= max_shift
        sub_lags, sub_corr = lags[mask], corr[mask]
        peak_i = int(np.argmax(sub_corr))
        best = float(sub_lags[peak_i])
        if 0 < peak_i < len(sub_corr) - 1:
            y0_, y1_, y2_ = sub_corr[peak_i - 1], sub_corr[peak_i], sub_corr[peak_i + 1]
            denom = y0_ - 2 * y1_ + y2_
            if denom != 0:
                best += 0.5 * (y0_ - y2_) / denom
        offs_used.append(off)
        shifts.append(best)

    if len(offs_used) < 4:
        raise RuntimeError(
            f"only {len(offs_used)} usable row slices to measure tilt "
            f"(need >= 4)"
        )

    offs_used = np.array(offs_used, dtype=float)
    shifts = np.array(shifts, dtype=float)
    coeffs, mask_good = polyfit_reject(offs_used, shifts, deg=1, thres=(5, 3))
    slope = float(coeffs[0])
    resid = shifts[mask_good] - np.polyval(coeffs, offs_used[mask_good])
    rms_px = float(np.sqrt(np.mean(resid ** 2))) if np.any(mask_good) else np.nan

    return TiltModel(slope, rms_px)


def _rectify_rows(frame2d, trace, tilt, y_lo, y_hi):
    """
    Resample rows y_lo:y_hi so each is de-sheared along the dispersion axis
    according to `tilt`, evaluated relative to `trace`'s own (slowly
    x-varying) centre. One np.interp call per row, vectorised across all
    columns - cheap (a handful of rows per fibre).

    Returns a (y_hi-y_lo, nx) array. If tilt is None, returns the
    corresponding frame2d slice unchanged (fibre with no measurable tilt).
    """
    ny, nx = frame2d.shape
    if tilt is None:
        return frame2d[y_lo:y_hi, :].copy()

    x_grid = np.arange(nx, dtype=float)
    y0_all = trace.y_of_x(x_grid)
    rectified = np.empty((y_hi - y_lo, nx), dtype=float)
    for i, row in enumerate(range(y_lo, y_hi)):
        dy = row - y0_all
        shift = tilt.shift_of_dy(dy)
        rectified[i, :] = np.interp(x_grid + shift, x_grid, frame2d[row, :])
    return rectified


def _aperture_half_width(x, trace, other_traces, times_sigma, margin=2.0):
    """Aperture half-width at column x, capped so it can't reach past the
    midpoint to the nearest other trace (minus a small margin)."""
    half = times_sigma * trace.sigma
    if not other_traces:
        return half
    y0 = trace.y_of_x(x)
    for other in other_traces:
        gap = abs(other.y_of_x(x) - y0) / 2 - margin
        half = min(half, max(gap, 1.0))
    return half


def extract_aperture(frame2d, trace, other_traces=None, tilt=None,
                      times_sigma=4.0, bg_gap=3, bg_width=10,
                      gain=1.0, reject_cosmics=True, cr_nsigma=8.0,
                      profile_bin_width=40, profile_poly_deg=2,
                      min_profile_snr=8.0, saturated_mask=None):
    """
    Extract a 1D spectrum along `trace` via two-pass optimal extraction.

    other_traces: TraceModel list for any sibling fibres, so the aperture
    and background bands are capped to never reach into a neighbour's
    light (see _aperture_half_width) - real overlap risk on this
    instrument's fibre spacing at generous times_sigma.

    tilt: TiltModel for this fibre, or None if untilted/unmeasured. When
    given, rows are de-sheared (_rectify_rows) before either extraction
    pass.

    saturated_mask: boolean array, same shape as frame2d, True where the
    *raw* (pre dark-subtraction) pixel hit the detector's saturation
    ceiling - pass this in rather than checking frame2d itself, since
    dark-subtraction and tilt-rectification (interpolation) both change
    pixel values and would mask a saturated value. Any column whose
    aperture window contains a saturated pixel gets flux=NaN: a saturated
    pixel isn't a recoverable outlier like a cosmic ray, the true flux
    there is simply unknown, and it's the observer's responsibility not to
    overexpose - this pipeline doesn't try to guess a value.

    Pass 1 uses fixed Gaussian weights (no CR rejection) to get a rough
    per-column flux S0(x) and background(x). That's used to build an
    empirical spatial profile P(dy, x): for each row offset dy from the
    trace centre, (row - background)/S0 is fit as a smooth low-order
    polynomial in x across all columns with enough S/N to constrain it -
    the true profile shape (boxcar-convolved-with-PSF, not Gaussian) and
    its slow drift along the dispersion axis fall out of the data rather
    than being assumed. Pass 2 re-extracts with P as the weights, flagging
    any pixel that doesn't fit S0(x)*P(dy,x) + background beyond cr_nsigma
    as a cosmic ray.

    Returns a dict with:
      flux           background-subtracted, profile-weighted flux per
                     column, cosmic-ray-cleaned: the matched-filter
                     estimate sum(P_i*(D_i-bg)) / sum(P_i**2) over
                     surviving pixels (the unbiased estimator for weights
                     P_i that represent the *fractional* PSF coverage per
                     pixel, i.e. sum(P_i)=1 - dividing by sum(P_i) instead
                     of sum(P_i**2) underestimates flux by ~sum(P_i**2),
                     see history above)
      weighted_raw   sum(P_i**2 * D_i_raw) / sum(P_i**2)**2 per column -
                     the quantity noise.py needs (with sumw2) for
                     CCD-equation error propagation on the *raw* (pre
                     background subtraction) counts, matching the flux
                     normalisation above.
      sumw2          sum(P_i**2) / sum(P_i**2)**2 = 1/sum(P_i**2) per column
      background     per-pixel local background level per column
      cr_rejected    number of pixels flagged as cosmic rays, per column
      profile        the fitted P(dy, x) as a dict {dy: poly1d}, for
                     diagnostics
    """
    ny, nx = frame2d.shape
    other_traces = other_traces or []

    # widest possible window across all columns, for rectification range
    x_grid = np.arange(nx)
    half_all = np.array([_aperture_half_width(x, trace, other_traces, times_sigma)
                          for x in range(0, nx, max(1, nx // 50))])
    half_max = float(np.max(half_all)) if len(half_all) else times_sigma * trace.sigma
    y0_all = trace.y_of_x(x_grid)
    y_lo_g = max(0, int(np.floor(np.min(y0_all) - half_max - bg_gap - bg_width)))
    y_hi_g = min(ny, int(np.ceil(np.max(y0_all) + half_max + bg_gap + bg_width)) + 1)
    if y_hi_g - y_lo_g < 3:
        raise RuntimeError("aperture window degenerate (trace too close to frame edge)")

    rect = _rectify_rows(frame2d, trace, tilt, y_lo_g, y_hi_g)

    # ---- pass 1: quick fixed-Gaussian extraction ----
    flux0 = np.full(nx, np.nan)
    bg0 = np.full(nx, np.nan)
    win0 = {}  # x -> (y_pixels, weights, y_lo, y_hi)
    for x in range(nx):
        y0 = trace.y_of_x(x)
        half = _aperture_half_width(x, trace, other_traces, times_sigma)
        y_lo = max(y_lo_g, int(np.floor(y0 - half)))
        y_hi = min(y_hi_g, int(np.ceil(y0 + half)) + 1)
        if y_hi - y_lo < 3:
            continue
        y_pixels = np.arange(y_lo, y_hi)
        weights = gaussian_pixel_weights(y_pixels, y0, trace.sigma)

        bg_lo = max(y_lo_g, y_lo - bg_gap - bg_width)
        bg_lo_hi = max(y_lo_g, y_lo - bg_gap)
        bg_hi_lo = min(y_hi_g, y_hi + bg_gap)
        bg_hi = min(y_hi_g, y_hi + bg_gap + bg_width)
        bg_pixels = np.concatenate([np.arange(bg_lo, bg_lo_hi),
                                    np.arange(bg_hi_lo, bg_hi)]).astype(int)
        if len(bg_pixels):
            bg_vals = rect[bg_pixels - y_lo_g, x]
            bg_level = float(np.median(bg_vals))
        else:
            bg_level = 0.0

        raw = rect[y_pixels - y_lo_g, x]
        # matched-filter flux estimate: for weights that represent the
        # *fractional* PSF coverage per pixel (sum(weights)=1), the
        # unbiased flux estimator is sum(w*(raw-bg))/sum(w**2), not
        # sum(w*(raw-bg))/sum(w) - see extract_aperture docstring/history
        # for why using the wrong one here corrupted the CR test badly.
        flux0[x] = np.sum(weights * (raw - bg_level)) / np.sum(weights ** 2)
        bg0[x] = bg_level
        win0[x] = (y_pixels, weights, y_lo, y_hi)

    # ---- build empirical profile P(dy, x) ----
    dy_min = int(np.floor(-half_max))
    dy_max = int(np.ceil(half_max))
    dys = np.arange(dy_min, dy_max + 1)
    profile_polys = {}
    finite = flux0[np.isfinite(flux0)]
    noise_floor = mad_std(finite) if len(finite) > 5 else 1.0
    good_cols = np.where(np.isfinite(flux0) & (flux0 > min_profile_snr * max(noise_floor, 1.0)))[0]
    if len(good_cols) < 20:
        good_cols = np.where(np.isfinite(flux0) & (flux0 > 0))[0]

    for dy in dys:
        xs_dy, ratios = [], []
        for x in good_cols:
            row = int(round(trace.y_of_x(x))) + dy
            if row < y_lo_g or row >= y_hi_g:
                continue
            val = rect[row - y_lo_g, x]
            ratios.append((val - bg0[x]) / flux0[x])
            xs_dy.append(x)
        if len(xs_dy) < profile_poly_deg + 2:
            profile_polys[dy] = None
            continue
        xs_dy = np.array(xs_dy, dtype=float)
        ratios = np.array(ratios, dtype=float)
        deg = min(profile_poly_deg, max(1, len(xs_dy) // 20))
        coeffs, mask_good = polyfit_reject(xs_dy, ratios, deg=deg, thres=(6, 4, 3))
        profile_polys[dy] = np.poly1d(coeffs)

    def profile_at(x, y_pixels, y0):
        dy_pix = np.round(y_pixels - y0).astype(int)
        vals = np.zeros(len(y_pixels))
        for i, dy in enumerate(dy_pix):
            poly = profile_polys.get(int(dy))
            vals[i] = max(poly(x), 0.0) if poly is not None else 0.0
        s = np.sum(vals)
        if s <= 0:
            return None
        return vals / s

    # ---- pass 2: profile-weighted extraction with CR rejection ----
    flux = np.full(nx, np.nan)
    weighted_raw = np.full(nx, np.nan)
    sumw2 = np.full(nx, np.nan)
    background = np.full(nx, np.nan)
    cr_rejected = np.zeros(nx, dtype=int)

    for x, (y_pixels, gweights, y_lo, y_hi) in win0.items():
        bg_level = bg0[x]
        raw = rect[y_pixels - y_lo_g, x]
        y0 = trace.y_of_x(x)

        P = profile_at(x, y_pixels, y0)
        if P is None:
            P = gweights  # fall back to the fixed-shape weights

        good = np.ones(len(y_pixels), dtype=bool)
        if reject_cosmics:
            # matched-filter flux estimate from P (see the sum(w**2) note
            # on flux0 above - same reasoning applies here). Using the
            # weighted-mean form (/sum(P), not /sum(P**2)) as the CR
            # reference systematically under-predicted every real profile
            # pixel by roughly sum(P**2) - which is well below 1 for a
            # profile spread over many pixels - and flagged nearly all of
            # them as spikes (regression-tested: this broke the July
            # single-fibre case badly before this fix).
            #
            # Both directions are tested: a cosmic ray spikes positive, but
            # a fixed dead/hot detector pixel reads anomalously *low* just
            # as often - found a real one this way (a single pixel reading
            # -11000 counts against a +-50 count local background, present
            # in every sub-exposure so it summed coherently in a coadd and
            # dominated the whole spectrum at that wavelength).
            #
            # Iterated (loose threshold first, then cr_nsigma): f0 computed
            # from *all* pixels is itself contaminated by whatever spike
            # it's supposed to help find, which biases predicted() for
            # every other pixel and was flagging legitimate bright profile
            # pixels near a real spike as collateral damage (regression-
            # tested: a single injected spike caused 5-6 pixels to be
            # rejected, not 1). Excluding obvious outliers first and
            # recomputing f0 from the survivors removes that bias.
            mask = np.ones(len(y_pixels), dtype=bool)
            for nsigma in (3 * cr_nsigma, cr_nsigma):
                w2 = np.sum(P[mask] ** 2)
                if w2 <= 0:
                    break
                f0 = np.sum(P[mask] * (raw[mask] - bg_level)) / w2
                predicted = f0 * P + bg_level
                noise = np.sqrt(np.maximum(np.maximum(predicted, bg_level), 1.0))
                spike = np.abs(raw - predicted) > nsigma * noise
                if np.any(spike) and np.sum(~spike) >= 3:
                    mask = ~spike
            if np.sum(~mask) and np.sum(mask) >= 3:
                good = mask
                cr_rejected[x] = int(np.sum(~mask))

        w = P[good]
        w2_sum = np.sum(w ** 2)
        if w2_sum <= 0:
            continue
        r = raw[good]
        flux[x] = np.sum(w * (r - bg_level)) / w2_sum
        weighted_raw[x] = np.sum((w ** 2) * r) / w2_sum ** 2
        sumw2[x] = np.sum(w ** 2) / w2_sum ** 2
        background[x] = bg_level

    saturated = np.zeros(nx, dtype=bool)
    if saturated_mask is not None:
        for x, (y_pixels, _, y_lo, y_hi) in win0.items():
            if np.any(saturated_mask[y_pixels, x]):
                saturated[x] = True
                flux[x] = np.nan
                weighted_raw[x] = np.nan
                sumw2[x] = np.nan

    return {
        "flux": flux,
        "weighted_raw": weighted_raw,
        "sumw2": sumw2,
        "background": background,
        "cr_rejected": cr_rejected,
        "saturated": saturated,
        "profile": profile_polys,
        "gain": gain,
    }
