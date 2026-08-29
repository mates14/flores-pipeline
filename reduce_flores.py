#!/usr/bin/env python3
"""
FLORES reduction driver: raw frames in, (wavelength, flux, error) triplets
out. No interactive steps.

Run from inside a night directory - the pipeline's convention is that the
night/run directory is the toplevel, and every stage's input/output lives
in a fixed subdirectory of it, so `cd` there and just run:

    cd /images/2026/20260823   # or ~/tmp/flores/20260823, wherever a given
                                # site keeps a night - this dir IS the root
    python3 /path/to/reduce_flores.py

    20260823/
    |-- raw/                 raw acquired frames (science/dark/comp) - input
    |-- sets/                manifests (comp, darks, standards, per-target
    |                        coadd lists) - see coadd_sets.py
    |-- coadded/             coadd_sets.py output (optional stage)
    |-- reduced/             <- this script's output
    `-- fluxcal/             flux_calibrate.py output

With no arguments this reads raw/ and writes reduced/ - and, if ./coadded
exists (coadd_sets.py has been run), its frames are automatically pulled
in too as additional science input, calibrated against the same raw/
comps/darks (a coadded product carries no comp/dark of its own - see
coadd_light.py). This is what makes coadded targets "just show up" without
symlinking them into raw/ (which would make them indistinguishable from a
raw frame, e.g. eligible for dark-subtraction they've already had - see
DARKSUB handling below) or hunting down --comp-dir. Point --coadd-dir
elsewhere, or pass '' to skip it, for a raw_dir that doesn't follow this
layout:

    python3 reduce_flores.py <raw_dir> [--out <out_dir>] [--seed seeds/flores_seed_linelist.csv]
    python3 reduce_flores.py <raw_dir> --comps comp1.fits comp2.fits   # legacy/un-migrated nights

--out always defaults to ./reduced regardless of raw_dir, since a night
normally reduces both raw/ (direct shots) and coadded/ (combined sets) into
the same reduced/ directory.

One raw frame in, one output spectrum out - always. This driver does no
co-adding/combining of its own: not every target is a single shot (some are
time-resolved series that must stay separate), and silently guessing which
frames "belong together" from headers alone is exactly the kind of fragile
heuristic this pipeline avoids elsewhere (see frames.py's comp-classification
history). If you want to combine several sub-exposures of one target (e.g.
a series of 5-minute shots instead of one long exposure), do that as an
explicit separate step (see coadd_light.py) and feed the resulting FITS
file in here like any other light frame - this driver doesn't care whether
a 'light' frame is a single raw exposure or an externally pre-combined
stack, only that its IMAGETYP/EXPTIME/DATE-OBS/BINNING headers are correct.

Frame roles come from IMAGETYP: dark / calib (ThAr comp) / light (or the
legacy 'object') - see frames.py. Low-res prism spectrographs like FLORES
hold their wavelength solution for as long as the input mirror isn't moved,
so comps are session-level anchors (typically one at the start of the
night, one at the end - not one per science frame the way an echelle setup
often needs). Accordingly:

FLORES feeds two fibres (two sky positions) into every raw frame - see
trace.py's module docstring for how they're found/told apart (one can show
a real geometric tilt, one usually doesn't; that's the authoritative tag,
not row order). Object/sky assignment is the *observer's* choice per
observation and isn't inferred - both fibres are always reduced and
written out separately, clearly labelled, and it's on the user to decide
downstream which is which for a given frame (matches this driver's
existing philosophy of not guessing what the user meant - see the
co-adding note above).

  1. classify frames (frames.py); pass --comps to name comp frames
     explicitly for nights where IMAGETYP doesn't mark them (older data).
  2. each comp frame is traced (both fibres), tilt-measured per fibre, and
     wavelength-calibrated ONCE per fibre (trace.py + wavecal.py),
     independent of any particular science frame. Comp solutions are
     cross-checked *within each fibre* across comps, and a warning is
     printed if a fibre's solutions disagree beyond expectation - that
     would mean the stable-calibration assumption didn't hold this night.
  3. each science frame is traced fresh (target position can drift during
     the night even if the wavelength solution doesn't - not every fibre
     is necessarily detectable on every science frame, e.g. a faint sky
     fibre may simply not show a traceable signal on a given exposure, in
     which case only the fibre/fibres actually found are reduced), is
     paired to its nearest-in-time comp, and each found fibre is matched
     to that comp's same fibre (by trace row position) and calibrated
     with its solution and tilt correction.
  4. error is propagated (noise.py) and barycentric correction applied if
     the frame has the needed headers.
  5. write <out_dir>/<science_stem>_fiber-<tag>.dat  (wavelength flux error)
     per fibre found, tag being e.g. "tilted"/"untilted".

A frame/fibre that fails trace-finding or wavelength-calibration's quality
gates is skipped with the reason printed to stderr, not silently emitted.
"""
import argparse
import glob
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.io import fits

import frames
import trace as tracemod
import wavecal
import noise as noisemod

SPEED_OF_LIGHT_KM_S = 299792.458


def get_barycorr(header):
    """
    Barycentric radial-velocity correction (km/s) for this frame, or None if
    the needed headers aren't present. Same approach as
    perek_pipelines/echelle_reduction.py:get_barycorr.
    """
    hreq = ["LATITUDE", "LONGITUD", "HEIGHT", "RA", "DEC", "DATE-OBS", "UT"]
    if not all(header.get(k) not in (None, "") for k in hreq):
        return None
    location = EarthLocation(
        lat=header["LATITUDE"], lon=header["LONGITUD"], height=header["HEIGHT"]
    )
    coord = SkyCoord(ra=header["RA"], dec=header["DEC"], unit=(u.hourangle, u.deg))
    otime = Time(
        header["DATE-OBS"] + "T" + header["UT"], format="isot", scale="utc",
        location=location,
    )
    return coord.radial_velocity_correction(obstime=otime).to(u.km / u.s).value


def wlshift(wl, vel_corr_km_s):
    return wl / (1 - (vel_corr_km_s / SPEED_OF_LIGHT_KM_S))


def solve_comp_fibers(comp, master_dark, seed_px, seed_wl, thar_wl, thar_intensity,
                      max_pixel_rms=5.0):
    """
    Trace + tilt-measure + wavelength-calibrate every fibre found in one
    comp frame. Returns {tag: {"trace":TraceModel, "tilt":TiltModel,
    "solution":DispersionSolution}}. A fibre that fails tilt measurement
    falls back to a zero-tilt model (most fibres are untilted anyway - see
    trace.py); one that fails wavecal's quality gate is omitted with a
    reason printed, not the whole comp.

    max_pixel_rms defaults looser here (5.0) than wavecal.solve_dispersion's
    own default (1.3, tuned for same-session self-consistency - see its
    validation in wavecal.py). A comp taken on a different night than the
    seed lands at ~1-6px rms even after coarse re-registration: real
    sub-pixel-scale drift beyond a single global shift, and in at least one
    case here, comp-lamp saturation on the brightest lines (a 120s exposure
    peaking at the same ~50k ADU as the 60s ones). 5px still rejects clear
    registration failures (a mis-locked coarse shift showed up at ~13px)
    without discarding every geometrically-valid comp from a different
    night.

    Raises RuntimeError if no fibre could even be traced.
    """
    comp_data = frames.subtract_dark(
        fits.getdata(comp.path).astype(float), master_dark
    )
    traces = tracemod.find_traces(comp_data)

    result = {}
    for i, tr in enumerate(traces):
        other = [t for j, t in enumerate(traces) if j != i]
        try:
            tilt = tracemod.measure_tilt(comp_data, tr)
        except RuntimeError:
            tilt = tracemod.TiltModel(0.0, np.nan)
        tr.tag = "tilted" if tilt.is_tilted() else "untilted"
        tag = tr.tag
        suffix = 1
        while tag in result:
            suffix += 1
            tag = f"{tr.tag}{suffix}"

        try:
            ext = tracemod.extract_aperture(
                comp_data, tr, other_traces=other,
                tilt=(tilt if tilt.is_tilted() else None),
            )
            solution = wavecal.solve_dispersion(
                ext["flux"], seed_px, seed_wl,
                thar_wl=thar_wl, thar_intensity=thar_intensity,
                max_pixel_rms=max_pixel_rms,
            )
        except RuntimeError as e:
            print(f"  fiber {tag} of comp {comp.path}: {e}", file=sys.stderr)
            continue
        result[tag] = {"trace": tr, "tilt": tilt, "solution": solution}

    return result


def check_comp_consistency(comp_fibers, max_median_shift_pix=1.0):
    """
    Compare comp solutions pairwise, *within each fibre tag* (a tilted
    fibre's solution is never compared against an untilted one). Returns a
    list of warning strings - empty if every fibre's typical (median)
    disagreement across comps is within max_median_shift_pix. See the
    single-fibre version this replaces for why median, not max, is the
    right statistic (module history).

    comp_fibers: {comp_path: {tag: {"solution":..., ...}}}
    """
    by_tag = {}
    for comp_path, fibers in comp_fibers.items():
        for tag, info in fibers.items():
            by_tag.setdefault(tag, {})[comp_path] = info["solution"]

    warnings = []
    for tag, solutions in by_tag.items():
        items = list(solutions.items())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (path_a, sol_a), (path_b, sol_b) = items[i], items[j]
                lo = max(sol_a.px_min, sol_b.px_min)
                hi = min(sol_a.px_max, sol_b.px_max)
                if lo >= hi:
                    warnings.append(f"[{tag}] {path_a} vs {path_b}: no pixel overlap to compare")
                    continue
                px = np.linspace(lo, hi, 300)
                wl_a = sol_a.pixel_to_wavelength(px)
                diff = np.abs(sol_b.wavelength_to_pixel(wl_a) - px)
                median_shift = np.nanmedian(diff)
                p90_shift = np.nanpercentile(diff, 90)
                max_shift = np.nanmax(diff)
                if not np.isfinite(median_shift) or median_shift > max_median_shift_pix:
                    warnings.append(
                        f"[{tag}] {path_a} vs {path_b}: solutions disagree by "
                        f"{median_shift:.2f} px (median), {p90_shift:.2f} px (p90), "
                        f"{max_shift:.2f} px (max) - check whether the input "
                        f"mirror moved"
                    )
    return warnings


def reduce_science_fibers(science, comp_fibers, master_dark, max_row_match_px=10.0,
                          saturation_adu=65535):
    """
    Trace every fibre visible on `science` (not every fibre need be
    detectable - e.g. a faint sky fibre may show no traceable signal on a
    given exposure), match each to the nearest comp fibre by trace row
    position, and extract + calibrate it with that fibre's solution and
    tilt correction.

    comp_fibers: this science frame's paired comp's per-fibre dict, as
    returned by solve_comp_fibers.

    A saturated raw pixel (>= saturation_adu, checked on the raw frame
    before dark-subtraction - see extract_aperture's saturated_mask
    docstring) makes that column's flux/error NaN in the output rather than
    a silently-wrong number: overexposure isn't something this pipeline
    tries to correct for, it's on the observer to not saturate the target.
    The NaN rows are kept (not dropped) so a gap is visible where the
    wavelength itself is still valid - only pixels with no wavelength
    assignment at all (outside the calibrated range) are dropped.

    If `science` is a coadd_light.py/coadd_sets.py product, it carries its
    own SATMASK extension (saturation checked per sub-exposure before
    summing) and that's used directly - checking the *combined* pixel
    values against saturation_adu would be wrong, since sub-exposures are
    summed, not averaged: a legitimately bright combined pixel can exceed
    a single frame's ADC ceiling without anything having saturated, and a
    pixel that did saturate in only one of several sub-exposures can sum
    to something under the ceiling and go undetected (found both ways on
    real data - see coadd_light.py's docstring for the numbers).

    Returns {tag: (wl, flux, error, err_method)} for each science trace
    successfully matched and reduced. Raises RuntimeError only if no trace
    at all could be found (per-fibre failures are just omitted from the
    returned dict, with a reason printed).
    """
    with fits.open(science.path) as hdul:
        raw_data = hdul[0].data.astype(float)
        if "SATMASK" in hdul:
            saturated_mask = hdul["SATMASK"].data.astype(bool)
        else:
            saturated_mask = raw_data >= saturation_adu
    sci_data = frames.subtract_dark(raw_data, master_dark)
    sci_traces = tracemod.find_traces(sci_data)

    x_ref = sci_data.shape[1] / 2
    results = {}
    for i, tr in enumerate(sci_traces):
        best_tag, best_dist = None, None
        for tag, info in comp_fibers.items():
            d = abs(info["trace"].y_of_x(x_ref) - tr.y_of_x(x_ref))
            if best_dist is None or d < best_dist:
                best_tag, best_dist = tag, d
        if best_tag is None or best_dist > max_row_match_px:
            print(f"  {science.path}: trace at y~{tr.y_of_x(x_ref):.1f} doesn't match "
                  f"any comp fibre within {max_row_match_px}px - skipped", file=sys.stderr)
            continue

        info = comp_fibers[best_tag]
        other = [t for j, t in enumerate(sci_traces) if j != i]
        try:
            ext = tracemod.extract_aperture(
                sci_data, tr, other_traces=other,
                tilt=(info["tilt"] if info["tilt"].is_tilted() else None),
                saturated_mask=saturated_mask,
            )
        except RuntimeError as e:
            print(f"  {science.path} fiber {best_tag}: {e}", file=sys.stderr)
            continue

        flux = ext["flux"]
        error, err_method = noisemod.estimate_error(ext, header=science.header)
        error[ext["saturated"]] = np.nan
        n_sat = int(np.sum(ext["saturated"]))
        if n_sat:
            print(f"  {science.path} fiber {best_tag}: {n_sat} columns saturated "
                  f"(>= {saturation_adu} ADU raw) - set to NaN", file=sys.stderr)

        pixels = np.arange(len(flux))
        wl = info["solution"].pixel_to_wavelength(pixels)
        valid = np.isfinite(wl)  # keep NaN flux/error (e.g. saturated) - only
                                  # drop pixels with no wavelength at all
        wl, flux, error = wl[valid], flux[valid], error[valid]

        radvel = get_barycorr(science.header)
        if radvel is not None:
            wl = wlshift(wl, radvel)

        order = np.argsort(wl)
        results[best_tag] = (wl[order], flux[order], error[order], err_method)

    return results


def _expand_manifest_arg(entries, raw_dir):
    """
    --comps/--darks accept either explicit FITS filenames, or a single
    manifest file listing one filename per line (blank lines and '#'
    comments ignored) - the natural way to point at a night's `sets/comp`
    or `sets/darks` file rather than typing out every frame on the command
    line. A single entry is treated as a manifest if it isn't itself a
    .fits/.fit file and resolves to a readable file (checked relative to
    raw_dir first, then as given - a manifest under sets/ is given as a
    path like "sets/comp", not found under raw_dir, so falls through to
    being read as given).

    Only the first tab-separated field of each line is used, so a manifest
    can be either a bare filename-per-line list (sets/comp's format) or the
    same tab-separated (filename, exptime, imagetyp) list-building tool
    format the per-target set lists use (sets/darks is written that way in
    practice) - both resolve to just the filenames.
    """
    if len(entries) != 1 or entries[0].lower().endswith((".fits", ".fit")):
        return entries

    candidate = entries[0]
    for path in (os.path.join(raw_dir, candidate), candidate):
        if os.path.isfile(path):
            with open(path) as f:
                lines = [ln.split("\t", 1)[0].strip() for ln in f]
            return [ln for ln in lines if ln and not ln.startswith("#")]

    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", nargs="?", default="raw",
                     help="directory of raw FITS frames for one night "
                          "(default: ./raw - see module docstring for the "
                          "assumed night-directory layout)")
    ap.add_argument("--out", default="reduced",
                     help="output dir (default: ./reduced, independent of "
                          "raw_dir - both raw/ and coadded/ passes normally "
                          "write into the same reduced/)")
    ap.add_argument("--sets-dir", default="sets",
                     help="directory holding this night's manifests - comp/darks/ "
                          "standards (default: ./sets). Used only to locate the "
                          "default --comps/--darks manifest when those aren't "
                          "given explicitly")
    ap.add_argument("--seed", default=os.path.join(
        os.path.dirname(__file__), "seeds", "flores_seed_linelist.csv"))
    ap.add_argument("--thar-atlas", default=os.path.join(
        os.path.dirname(__file__), "thar_lovis_pepe_clean.csv"))
    ap.add_argument("--comps", nargs="+", default=None,
                     help="explicit comp frame filename(s), OR a single manifest "
                          "file listing one filename per line, overriding "
                          "header-based classification (recommended - see module "
                          "docstring). Default: <sets-dir>/comp if that file "
                          "exists, else header-based classification")
    ap.add_argument("--darks", nargs="+", default=None,
                     help="explicit raw dark frame filename(s), OR a single "
                          "manifest file listing one filename per line, "
                          "overriding header-based classification for nights "
                          "where darks aren't tagged correctly (same convention "
                          "as --comps). Default: <sets-dir>/darks if that file "
                          "exists, else header-based classification")
    ap.add_argument("--comp-dir", default=None,
                     help="directory --comps/--darks entries/manifests are "
                          "resolved relative to (default: raw_dir). Set this "
                          "when raw_dir holds coadded frames (see coadd_sets.py) "
                          "that don't have their own comps/darks - point it at "
                          "raw/ instead, e.g. "
                          "reduce_flores.py coadded --comp-dir raw")
    ap.add_argument("--coadd-dir", default="coadded",
                     help="directory of coadd_sets.py output, automatically "
                          "merged in as additional science input alongside "
                          "raw_dir when it exists (default: ./coadded; pass '' "
                          "to skip). Never a source of comp/dark frames - those "
                          "still come only from raw_dir/--comp-dir, since a "
                          "coadded product carries none of its own. Ignored if "
                          "it resolves to the same directory as raw_dir")
    ap.add_argument("--max-comp-gap-min", type=float, default=None,
                     help="hard cutoff on comp-to-science time gap; default is "
                          "unlimited (comps are session-level anchors)")
    ap.add_argument("--max-wavecal-rms", type=float, default=5.0,
                     help="quality gate on a comp's dispersion fit, in pixels "
                          "(default 5.0 - see solve_comp docstring for why "
                          "this is looser than wavecal.solve_dispersion's own "
                          "1.3px default)")
    ap.add_argument("--saturation-adu", type=float, default=65535,
                     help="raw ADU value (pre dark-subtraction) at/above which a "
                          "science pixel is treated as saturated and its column's "
                          "flux/error come out as NaN (default: the 16-bit ADC "
                          "ceiling - lower this if the camera's true full well "
                          "saturates before that)")
    args = ap.parse_args()

    raw_dir = os.path.normpath(args.raw_dir)
    comp_dir = os.path.normpath(args.comp_dir) if args.comp_dir else raw_dir
    out_dir = os.path.normpath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    if args.comps is None:
        default_comps = os.path.join(args.sets_dir, "comp")
        if os.path.isfile(default_comps):
            args.comps = [default_comps]
    if args.darks is None:
        default_darks = os.path.join(args.sets_dir, "darks")
        if os.path.isfile(default_darks):
            args.darks = [default_darks]

    seed_px, seed_wl, _ = wavecal.load_seed_linelist(args.seed)
    thar_wl, thar_intensity = wavecal.load_thar_atlas(args.thar_atlas)

    paths = sorted(glob.glob(os.path.join(raw_dir, "*.fits")))
    all_frames = frames.classify_all(paths)
    by_path = {f.path: f for f in all_frames}

    if comp_dir != raw_dir:
        comp_dir_paths = sorted(glob.glob(os.path.join(comp_dir, "*.fits")))
        comp_dir_frames = frames.classify_all(comp_dir_paths)
        by_path.update({f.path: f for f in comp_dir_frames})
    else:
        comp_dir_frames = all_frames

    science_frames = [f for f in all_frames if f.role == "science"]

    coadd_dir = os.path.normpath(args.coadd_dir) if args.coadd_dir else None
    if coadd_dir and coadd_dir != raw_dir and os.path.isdir(coadd_dir):
        coadd_paths = sorted(glob.glob(os.path.join(coadd_dir, "*.fits")))
        coadd_science = [f for f in frames.classify_all(coadd_paths) if f.role == "science"]
        by_path.update({f.path: f for f in coadd_science})
        science_frames = science_frames + coadd_science
        print(f"{len(coadd_science)} additional science frames found in {coadd_dir}",
              file=sys.stderr)

    if args.comps:
        comp_entries = _expand_manifest_arg(args.comps, comp_dir)
        comp_frames = []
        for c in comp_entries:
            c_path = c if os.path.isabs(c) else os.path.join(comp_dir, c)
            comp_frames.append(by_path.get(c_path, frames.classify(c_path)))
        science_frames = [f for f in science_frames if f.path not in
                           {c.path for c in comp_frames}]
    else:
        comp_frames = [f for f in comp_dir_frames if f.role == "comp"]
    mdark_frames = [f for f in all_frames if f.role == "mdark"]
    dark_frames = [f for f in all_frames if f.role == "dark"]
    if comp_dir != raw_dir:
        # comps (raw ThAr frames) need darks from their own directory too -
        # the coadded science frames don't (already DARKSUB=T), so this is
        # additive, not a replacement
        mdark_frames = mdark_frames + [f for f in comp_dir_frames if f.role == "mdark"]
        dark_frames = dark_frames + [f for f in comp_dir_frames if f.role == "dark"]
    if args.darks:
        # explicit manifest overrides header-based dark classification
        # entirely (same convention as --comps), for nights where raw darks
        # aren't tagged correctly - already-combined master darks (mdark,
        # unambiguous by construction) are left as header-classified
        dark_entries = _expand_manifest_arg(args.darks, comp_dir)
        dark_frames = []
        for d in dark_entries:
            d_path = d if os.path.isabs(d) else os.path.join(comp_dir, d)
            dark_frames.append(by_path.get(d_path, frames.classify(d_path)))
        science_frames = [f for f in science_frames if f.path not in
                           {d.path for d in dark_frames}]

    print(f"{len(science_frames)} science, {len(comp_frames)} comp, "
          f"{len(mdark_frames)} master dark, {len(dark_frames)} raw dark "
          f"frames found", file=sys.stderr)

    if not comp_frames:
        print("no comp frames identified - nothing can be wavelength-calibrated. "
              "Pass --comps if headers don't mark them.", file=sys.stderr)
        return

    dark_cache = {}

    def get_dark(f):
        if f.dark_subtracted:
            # already dark-subtracted per sub-exposure by coadd_light.py
            # (DARKSUB header) - subtracting again would double-count it
            return None
        key = (f.exptime, f.binning)
        if key not in dark_cache:
            dark_cache[key] = frames.get_master_dark(
                f.exptime, f.binning, mdark_frames, dark_frames
            )
        return dark_cache[key]

    # solve each comp exactly once, per fibre (they're session-level
    # anchors, not matched 1:1 with science frames)
    comp_fibers = {}
    for comp in comp_frames:
        try:
            fibers = solve_comp_fibers(
                comp, get_dark(comp), seed_px, seed_wl, thar_wl, thar_intensity,
                max_pixel_rms=args.max_wavecal_rms,
            )
        except RuntimeError as e:
            print(f"SKIP comp {comp.path}: {e}", file=sys.stderr)
            continue
        if fibers:
            comp_fibers[comp.path] = fibers
        else:
            print(f"SKIP comp {comp.path}: no fibre could be wavelength-calibrated",
                  file=sys.stderr)
    comp_frames = [c for c in comp_frames if c.path in comp_fibers]

    if not comp_frames:
        print("no comp frame could be wavelength-calibrated - aborting.",
              file=sys.stderr)
        return

    for warning in check_comp_consistency(comp_fibers):
        print(f"WARNING: {warning}", file=sys.stderr)

    n_ok, n_skip = 0, 0

    for sci in science_frames:
        comp = frames.find_paired_comp(sci, comp_frames,
                                        max_gap_minutes=args.max_comp_gap_min)
        if comp is None:
            print(f"SKIP {sci.path}: no comp frame available", file=sys.stderr)
            n_skip += 1
            continue

        try:
            fiber_results = reduce_science_fibers(
                sci, comp_fibers[comp.path], get_dark(sci),
                saturation_adu=args.saturation_adu,
            )
        except RuntimeError as e:
            print(f"SKIP {sci.path}: {e}", file=sys.stderr)
            n_skip += 1
            continue

        if not fiber_results:
            print(f"SKIP {sci.path}: no fibre could be matched/extracted", file=sys.stderr)
            n_skip += 1
            continue

        stem = os.path.splitext(os.path.basename(sci.path))[0]
        for tag, (wl, flux, error, err_method) in fiber_results.items():
            solution = comp_fibers[comp.path][tag]["solution"]
            out_path = os.path.join(out_dir, f"{stem}_fiber-{tag}.dat")
            header = (
                f"wavelength[Angstrom] flux error  "
                f"fiber={tag} comp={os.path.basename(comp.path)} "
                f"wavecal_rms_px={solution.rms_pix:.3f} n_lines={solution.n_lines} "
                f"error_method={err_method}"
            )
            np.savetxt(out_path, np.column_stack([wl, flux, error]),
                       header=header, fmt="%.4f")
            print(f"OK   {sci.path} [{tag}] -> {out_path} "
                  f"(comp={os.path.basename(comp.path)}, rms={solution.rms_pix:.2f}px)",
                  file=sys.stderr)
        n_ok += 1

    print(f"done: {n_ok} reduced, {n_skip} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
