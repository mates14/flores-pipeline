#!/usr/bin/env python3
"""
Flux-calibrate a directory of reduce_flores.py output using
spectrophotometric standards - see fluxcal.py's module docstring for the
method and its limitations (no atmospheric extinction correction: no
frame in this dataset has airmass information).

    python3 flux_calibrate.py <reduced_dir> [--out <dir>] [--coadd-dir <dir>]

Derives one response curve from etauma/zetacas/vega/hr7596's already-
reduced spectra in <reduced_dir> (produced by the coadd_sets.py ->
reduce_flores.py workflow), then applies it to every .dat file in
<reduced_dir> - the four standards themselves (as a consistency check) and
everything else, including hd199478 (a real target, not a calibrator -
it's a known variable supergiant, not a flux standard, see fluxcal.py).

Each standard's .dat stem (e.g. "etauma_6x5s") is expected to match a FITS
file of the same name in --coadd-dir (default: <reduced_dir>/..), which is
where coadd_sets.py's coadd_light.py-produced frames carry their EXPTIME -
needed to convert the reduced (total-counts) spectrum to a count rate
before dividing by the reference flux.
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

import fluxcal

STANDARDS = {
    "etauma": ("fluxstd/etauma.tab", fluxcal.load_maestro_tab),
    "zetacas": ("fluxstd/zeta-cas.tab", fluxcal.load_maestro_tab),
    "vega": ("fluxstd/calspec_vega.fits", fluxcal.load_calspec),
    "hr7596": ("fluxstd/hr7596_spec16cal.dat", fluxcal.load_iraf_onedstd),
}


def load_reduced(path):
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1], d[:, 2]


def find_source_fits(dat_path, coadd_dir):
    """etauma_6x5s_fiber-untilted.dat -> <coadd_dir>/etauma_6x5s.fits"""
    stem = os.path.basename(dat_path)
    for tag_suffix in ("_fiber-tilted", "_fiber-untilted", "_fiber-tilted2",
                       "_fiber-untilted2"):
        idx = stem.find(tag_suffix)
        if idx != -1:
            stem = stem[:idx]
            break
    stem = stem[:-4] if stem.endswith(".dat") else stem
    return os.path.join(coadd_dir, stem + ".fits")


def target_name(dat_path):
    """etauma_6x5s_fiber-untilted.dat -> etauma"""
    base = os.path.basename(dat_path)
    return base.split("_")[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reduced_dir")
    ap.add_argument("--out", default=None,
                     help="output dir (default: <reduced_dir>/fluxcal)")
    ap.add_argument("--coadd-dir", default=None,
                     help="directory holding the coadded FITS files whose EXPTIME "
                          "headers are needed (default: <reduced_dir>/..)")
    args = ap.parse_args()

    reduced_dir = os.path.normpath(args.reduced_dir)
    coadd_dir = args.coadd_dir or os.path.join(reduced_dir, "..")
    out_dir = args.out or os.path.join(reduced_dir, "fluxcal")
    os.makedirs(out_dir, exist_ok=True)

    dat_paths = sorted(glob.glob(os.path.join(reduced_dir, "*.dat")))
    by_target = {}
    for p in dat_paths:
        by_target.setdefault(target_name(p), []).append(p)

    responses = []
    for name, (ref_path, loader) in STANDARDS.items():
        ref_path = os.path.join(os.path.dirname(__file__), ref_path)
        ref_wl, ref_flambda = loader(ref_path)

        dat_candidates = by_target.get(name, [])
        if not dat_candidates:
            print(f"SKIP standard {name}: no reduced spectrum found in {reduced_dir}",
                  file=sys.stderr)
            continue
        dat_path = dat_candidates[0]
        source_fits = find_source_fits(dat_path, coadd_dir)
        if not os.path.isfile(source_fits):
            print(f"SKIP standard {name}: source FITS not found ({source_fits})",
                  file=sys.stderr)
            continue
        exptime = float(fits.getheader(source_fits)["EXPTIME"])

        wl, flux, _ = load_reduced(dat_path)
        try:
            resp = fluxcal.derive_response_one(wl, flux, exptime, ref_wl, ref_flambda)
        except RuntimeError as e:
            print(f"SKIP standard {name}: {e}", file=sys.stderr)
            continue
        responses.append(resp)
        print(f"OK   standard {name}: response derived from {dat_path} "
              f"(exptime={exptime:.1f}s, valid {resp.wl_min:.0f}-{resp.wl_max:.0f}A)",
              file=sys.stderr)

    if not responses:
        print("no standard yielded a usable response curve - aborting.", file=sys.stderr)
        return

    combined = fluxcal.combine_responses(responses)
    print(f"combined response from {len(responses)} standards, "
          f"valid {combined.wl_min:.0f}-{combined.wl_max:.0f}A", file=sys.stderr)

    n_ok, n_skip = 0, 0
    for dat_path in dat_paths:
        name = target_name(dat_path)
        source_fits = find_source_fits(dat_path, coadd_dir)
        if not os.path.isfile(source_fits):
            print(f"SKIP {dat_path}: source FITS not found ({source_fits})",
                  file=sys.stderr)
            n_skip += 1
            continue
        exptime = float(fits.getheader(source_fits)["EXPTIME"])

        wl, flux, error = load_reduced(dat_path)
        rate = flux / exptime
        rate_err = error / exptime
        resp_here = combined(wl)
        valid = np.isfinite(resp_here) & (resp_here > 0)

        flam = np.full(len(wl), np.nan)
        flam_err = np.full(len(wl), np.nan)
        flam[valid] = rate[valid] / resp_here[valid]
        flam_err[valid] = rate_err[valid] / resp_here[valid]

        stem = os.path.splitext(os.path.basename(dat_path))[0]
        out_path = os.path.join(out_dir, f"{stem}_fluxcal.dat")
        header = (
            "wavelength[Angstrom] flux[erg/s/cm2/A] error[erg/s/cm2/A]  "
            f"response_from={'+'.join(STANDARDS.keys())} "
            f"response_range={combined.wl_min:.0f}-{combined.wl_max:.0f}A "
            "CAVEAT=no_atmospheric_extinction_correction_applied_no_airmass_in_headers"
        )
        np.savetxt(out_path, np.column_stack([wl, flam, flam_err]),
                   header=header, fmt="%.6e")
        print(f"OK   {dat_path} -> {out_path}", file=sys.stderr)
        n_ok += 1

    print(f"done: {n_ok} flux-calibrated, {n_skip} skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
