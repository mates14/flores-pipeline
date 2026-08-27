#!/usr/bin/env python3
"""
Coadd each target's "long" sub-exposures ahead of extraction, driven by a
per-night sets/ directory (see reduce_flores.py's module docstring for the
assumed night-directory layout - this script is the sets/+raw/ -> coadded/
stage). Run from inside the night directory:

    cd 20260823 && python3 /path/to/coadd_sets.py

which reads raw/*.fits, sets/list (one target-list filename per line, e.g.
`sscyg`, `vega`, ...) and each listed sets/<name> file, and writes
coadded/. All three are overridable and independent:

    python3 coadd_sets.py <raw_dir> [--out <coadd_dir>] [--sets <sets/list>] [--only target1 target2 ...]

Each per-target list file (sets/<name>, tab-separated: filename, exptime,
imagetyp - the format your own list-building tool already writes) is
grouped by exposure time. Frequently a target has more than one exposure
length in its list
(quick test shots, or an exploratory long one) - rather than take "short =
testing" as an absolute rule, this picks the LONGEST group that has at
least 2 members and coadds only that. sets/comp and sets/darks (comp/dark
manifests - see reduce_flores.py) live alongside the per-target lists but
have a different, special role and aren't science sets themselves; they're
skipped automatically here since they don't parse as a per-frame target
list, as is anything else dropped into sets/ that isn't listed in
sets/list at all (e.g. a master `img` index, not a single-target set).

Why "longest group with >=2 members" rather than just "the long ones":
plain "combine the longest exposures, drop the short ones" breaks on a set
like fb179 (three 120s + a single 300s) - a single long exposure isn't
something to usefully "coadd" alone, and it's the 120s group that has the
real multi-frame S/N gain. Checked against every target list in this
night's sets/list, this rule reproduces exactly what "ignore the short
test shots, combine the long ones" means in every case, including the
ones where that phrase alone is ambiguous (fb179; algol, which has both a
7x1s and a 4x20s group).

Each target's chosen group is coadded via coadd_light.py's coadd()
(dark-subtracted per sub-exposure, registered on trace position, summed -
see its docstring) and written to <coadd_dir>/<target>_<n>x<exptime>s.fits,
tagged DARKSUB=T exactly like coadd_light.py's own output, so it can be fed
straight into reduce_flores.py like any other light frame - point
reduce_flores.py's raw_dir at <coadd_dir> and --comp-dir at raw/ (the
coadded frames don't carry their own comps/darks - reduce_flores.py
doesn't need them there anyway, since the coadd is already
dark-subtracted and DARKSUB=T-tagged), e.g.:

    python3 reduce_flores.py coadded --comp-dir raw
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

import frames
import coadd_light


def parse_target_list(path):
    """
    Parse a tab-separated target list (filename, exptime, imagetyp, ...).
    Returns a list of (filename, exptime) tuples, or None if the file
    doesn't look like this format (e.g. a bare filename-per-line manifest
    like `comp`).
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                return None
            try:
                exptime = float(parts[1])
            except ValueError:
                return None
            rows.append((parts[0].strip(), exptime))
    return rows


def pick_group(rows, min_members=2):
    """
    Group (filename, exptime) rows by exptime; return (exptime, filenames)
    for the longest group with >= min_members, or (None, None) if no group
    qualifies.
    """
    by_exptime = {}
    for fn, et in rows:
        by_exptime.setdefault(et, []).append(fn)

    qualifying = [et for et, fns in by_exptime.items() if len(fns) >= min_members]
    if not qualifying:
        return None, None
    best_et = max(qualifying)
    return best_et, by_exptime[best_et]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir", nargs="?", default="raw",
                     help="directory of raw FITS frames (default: ./raw)")
    ap.add_argument("--sets", default="sets/list",
                     help="path to the sets index file, listing one target-list "
                          "filename per line (default: ./sets/list). Each listed "
                          "name is resolved as a file in the same directory as "
                          "this index (i.e. sets/<name>), not in raw_dir")
    ap.add_argument("--out", default="coadded",
                     help="output dir for coadds (default: ./coadded)")
    ap.add_argument("--only", nargs="+", default=None,
                     help="only process these target names (default: all in --sets)")
    ap.add_argument("--min-members", type=int, default=2,
                     help="minimum sub-exposures in a group to bother coadding (default 2)")
    ap.add_argument("--max-list-size", type=int, default=30,
                     help="skip a list with more entries than this - guards against "
                          "accidentally treating a night-wide index (e.g. `img`) as a "
                          "single target's sub-exposures (default 30)")
    ap.add_argument("--saturation-adu", type=float, default=65535,
                     help="raw ADU value (per sub-exposure, pre dark-subtraction) "
                          "at/above which a pixel is flagged saturated (default: "
                          "the 16-bit ADC ceiling) - see coadd_light.py's docstring "
                          "for why this must be checked per sub-exposure, not on "
                          "the summed total")
    args = ap.parse_args()

    raw_dir = os.path.normpath(args.raw_dir)
    sets_path = os.path.normpath(args.sets)
    sets_dir = os.path.dirname(sets_path) or "."
    out_dir = os.path.normpath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    with open(sets_path) as f:
        target_names = [ln.strip() for ln in f if ln.strip()]
    if args.only:
        target_names = [t for t in target_names if t in args.only]

    dark_paths = sorted(glob.glob(os.path.join(raw_dir, "*.fits")))
    dark_candidates = frames.classify_all(dark_paths)
    mdark_frames = [f for f in dark_candidates if f.role == "mdark"]
    dark_frames = [f for f in dark_candidates if f.role == "dark"]

    for name in target_names:
        list_path = os.path.join(sets_dir, name)
        if not os.path.isfile(list_path):
            print(f"SKIP {name}: no such list file ({list_path})", file=sys.stderr)
            continue
        rows = parse_target_list(list_path)
        if rows is None:
            print(f"SKIP {name}: not a per-frame target list (utility/manifest file)",
                  file=sys.stderr)
            continue
        if len(rows) > args.max_list_size:
            print(f"SKIP {name}: {len(rows)} entries exceeds --max-list-size "
                  f"{args.max_list_size} - looks like a night-wide index (e.g. `img`), "
                  f"not a single-target list", file=sys.stderr)
            continue

        exptime, filenames = pick_group(rows, min_members=args.min_members)
        if exptime is None:
            counts = {}
            for _, et in rows:
                counts[et] = counts.get(et, 0) + 1
            print(f"SKIP {name}: no exposure-length group with >= {args.min_members} "
                  f"members ({counts})", file=sys.stderr)
            continue

        sub_frames = sorted(
            (frames.classify(os.path.join(raw_dir, fn)) for fn in filenames),
            key=lambda f: f.mjd,
        )
        try:
            total, shifts, sat_mask = coadd_light.coadd(
                sub_frames, mdark_frames, dark_frames,
                saturation_adu=args.saturation_adu,
            )
        except RuntimeError as e:
            print(f"SKIP {name}: {e}", file=sys.stderr)
            continue
        header = coadd_light.build_header(sub_frames, shifts)
        out_path = os.path.join(out_dir, f"{name}_{len(sub_frames)}x{exptime:g}s.fits")
        primary = fits.PrimaryHDU(data=total.astype(np.float32), header=header)
        sat_hdu = fits.ImageHDU(data=sat_mask.astype(np.uint8), name="SATMASK")
        fits.HDUList([primary, sat_hdu]).writeto(out_path, overwrite=True)
        print(f"OK   {name}: {len(sub_frames)} x {exptime:g}s -> {out_path} "
              f"(total {header['EXPTIME']:.0f}s, y-shifts {shifts}, "
              f"{int(sat_mask.sum())} saturated px)", file=sys.stderr)


if __name__ == "__main__":
    main()
