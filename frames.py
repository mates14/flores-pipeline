"""
FITS frame classification and dark handling for FLORES.

Going forward, acquisitions tag IMAGETYP as dark / calib / light (calib =
ThAr comp lamp) - that convention resolves what used to be an unrecoverable
ambiguity (see git history: the 20260823 night had to be de-ambiguated by
hand via a `comp` manifest file, because everything was tagged 'object').
'object' and legacy comp-ish OBJECT substrings are still recognised for
older/un-migrated data. Flat-fielding is out of scope for now (no flats
exist in any dataset checked) - `apply_flat` is a no-op hook for later.
"""
import bisect
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.time import Time

# --- classification config -------------------------------------------------
DARK_IMAGETYPES = {"dark"}
MDARK_IMAGETYPES = {"mdark"}  # already-combined master darks, e.g. from proc_darks2
COMP_IMAGETYPES = {"calib", "comp", "comparison", "thar", "wave"}
COMP_OBJECT_SUBSTRINGS = ("thar", "comp", "arc")  # legacy fallback only
SCIENCE_IMAGETYPES = {"light", "object"}
FLAT_IMAGETYPES = {"flat", "flatfield"}
# ----------------------------------------------------------------------------


@dataclass
class Frame:
    path: str
    role: str  # "science" | "dark" | "mdark" | "comp" | "flat" | "unknown"
    exptime: float
    mjd: float
    binning: str
    dark_subtracted: bool
    header: fits.Header


def classify(path):
    header = fits.getheader(path)
    imagetyp = str(header.get("IMAGETYP", "")).strip().lower()
    obj = str(header.get("OBJECT", "") or "").strip().lower()

    if imagetyp in MDARK_IMAGETYPES:
        role = "mdark"
    elif imagetyp in DARK_IMAGETYPES:
        role = "dark"
    elif imagetyp in FLAT_IMAGETYPES:
        role = "flat"
    elif imagetyp in COMP_IMAGETYPES or any(s in obj for s in COMP_OBJECT_SUBSTRINGS):
        role = "comp"
    elif imagetyp in SCIENCE_IMAGETYPES:
        role = "science"
    else:
        role = "unknown"

    exptime = float(header.get("EXPTIME", header.get("EXPOSURE", np.nan)))
    date_obs = header.get("DATE-OBS")
    mjd = Time(date_obs, format="isot", scale="utc").mjd if date_obs else np.nan
    binning = str(header.get("BINNING", "") or
                  f"{header.get('BINX', '?')}x{header.get('BINY', '?')}")
    dark_subtracted = bool(header.get("DARKSUB", False))

    return Frame(path=path, role=role, exptime=exptime, mjd=mjd,
                 binning=binning, dark_subtracted=dark_subtracted, header=header)


def classify_all(paths):
    return [classify(p) for p in paths]


def get_master_dark(exptime, binning, mdark_frames, dark_frames, tol=0.5):
    """
    Master dark for `exptime`/`binning`. Prefers an already-combined mdark
    (e.g. proc_darks2 output, IMAGETYP=mdark) matched by exptime and
    binning; falls back to median-combining raw `dark` frames if no mdark
    matches. Returns None if neither is available.
    """
    for f in mdark_frames:
        if abs(f.exptime - exptime) <= tol and f.binning == binning:
            return fits.getdata(f.path).astype(float)

    matched = [f for f in dark_frames
               if abs(f.exptime - exptime) <= tol and f.binning == binning]
    if not matched:
        return None
    stack = np.stack([fits.getdata(f.path).astype(float) for f in matched])
    return np.median(stack, axis=0)


def subtract_dark(data, master_dark):
    if master_dark is None:
        return data
    return data - master_dark


def apply_flat(data, master_flat=None):
    """No-op for now - flat-fielding is out of scope until flats exist."""
    return data


def find_paired_comp(science_frame, comp_frames, max_gap_minutes=None):
    """
    Nearest-in-time comp frame to `science_frame`.

    Low-res prism spectrographs like FLORES hold their wavelength solution
    for as long as the input mirror isn't moved, so comps are session-level
    anchors (typically one at the start of the night, one at the end, not
    one per science frame) - unlike a per-target calibration set. So the
    default has no time-gap cutoff: pick whichever comp is nearest in time,
    even if that's hours away. Pass max_gap_minutes to reinstate a hard cutoff
    if that assumption doesn't hold for a particular night/instrument state.

    Returns None only if there's no comp frame at all (or none within an
    explicit max_gap_minutes) - callers should skip the science frame with a
    clear reason logged, not guess.
    """
    if not comp_frames or not np.isfinite(science_frame.mjd):
        return None
    mjds = sorted((f.mjd, f) for f in comp_frames if np.isfinite(f.mjd))
    if not mjds:
        return None
    keys = [m for m, _ in mjds]
    i = bisect.bisect_left(keys, science_frame.mjd)
    candidates = [mjds[j] for j in (i - 1, i) if 0 <= j < len(mjds)]
    if not candidates:
        return None
    best_mjd, best_frame = min(candidates, key=lambda kf: abs(kf[0] - science_frame.mjd))
    if max_gap_minutes is not None and abs(best_mjd - science_frame.mjd) * 24 * 60 > max_gap_minutes:
        return None
    return best_frame
