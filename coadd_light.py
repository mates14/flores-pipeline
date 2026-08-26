#!/usr/bin/env python3
"""
Combine several raw light sub-exposures of one target into a single frame -
for "a series of 5-minute shots instead of one risky long exposure".

    python3 coadd_light.py out.fits frame1.fits frame2.fits ...
    python3 coadd_light.py out.fits frame1.fits frame2.fits --dark-dir /images/2026/20260823

This is a separate, explicit, opt-in step - reduce_flores.py never calls it
automatically (see its module docstring for why: not every target is a
single shot, some are time-resolved series that must stay separate, and
guessing which frames "belong together" from headers is exactly the kind of
fragile heuristic this pipeline avoids elsewhere). Run this by hand when you
actually want a coadd, then feed the output to reduce_flores.py like any
other light frame.

Sub-exposures are registered on their cross-dispersion (Y) trace position
before summing (integer-pixel shift-and-add via trace.py's trace finder) -
seeing/guiding drift between sub-exposures would otherwise smear/broaden
the coadded trace. Since both FLORES fibres are fixed relative to each
other on the focal plane, guiding drift moves them together - one fibre is
traced as the registration reference and the whole frame (both fibres) is
rolled by that single shift, rather than tracking each fibre separately.
Dispersion (X)-axis drift is NOT corrected here: over the short span a
coadd sequence normally covers this is negligible (wavecal.py's
same-session comp checks agree to a fraction of a pixel - see its
validation), and any residual is absorbed by the wavelength calibration
step downstream anyway.

Each sub-exposure is dark-subtracted individually first (matched by its own
EXPTIME/BINNING, same lookup as reduce_flores.py uses), since a
proc_darks2-style dark ladder won't generally have an exact match for the
combined EXPTIME. The output is tagged DARKSUB=T so reduce_flores.py knows
not to subtract a dark again.

No cosmic-ray rejection - a straight sum, not a sigma-clipped combine. Fine
for a handful of sub-exposures; revisit if that stops being true.

Saturation is checked per sub-exposure, on its own raw pixel values, BEFORE
summing - not on the combined total. Sub-exposures are summed, not
averaged, so the combined value at a pixel can legitimately run to several
times the single-frame ADC ceiling without any individual exposure ever
having saturated there (checking the *sum* against that ceiling would
falsely flag good data); conversely a pixel that saturated in only one of
several sub-exposures can sum to something comfortably under the ceiling
and pass an after-the-fact check silently, with bad data baked in
undetected. The per-exposure saturation flags are OR'd together (registered
the same way the data is) and written as a second HDU, SATMASK, in the
output file - reduce_flores.py reads that directly for a coadd instead of
re-deriving (wrongly) from the combined pixel values.
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits

import frames
import trace as tracemod


def coadd(sub_frames, mdark_frames, dark_frames, saturation_adu=65535):
    """
    sub_frames: list of frames.Frame, all the same binning, sorted by time.
    Returns (summed_2d_array, per_frame_y_shifts, saturated_mask) - see
    module docstring for why saturation is tracked per sub-exposure rather
    than on the combined total.
    """
    binnings = {f.binning for f in sub_frames}
    if len(binnings) > 1:
        raise ValueError(f"inputs have mixed binning, can't combine: {binnings}")

    total = None
    sat_mask = None
    shifts = []
    ref_trace = None
    ref_x = None

    for f in sub_frames:
        raw = fits.getdata(f.path).astype(float)
        frame_sat = raw >= saturation_adu
        if f.dark_subtracted:
            data = raw  # already done upstream (e.g. re-coadding a coadd)
        else:
            md = frames.get_master_dark(f.exptime, f.binning, mdark_frames, dark_frames)
            data = frames.subtract_dark(raw, md)

        traces = tracemod.find_traces(data)
        if ref_trace is None:
            # first frame sets the registration reference fibre
            tr = traces[0]
            ref_trace = tr
            ref_x = (tr.x_min + tr.x_max) / 2
            shift = 0
        else:
            # match to the same fibre by nearest row position, in case a
            # spurious extra trace appears/disappears between frames
            tr = min(traces, key=lambda t: abs(t.y_of_x(ref_x) - ref_trace.y_of_x(ref_x)))
            shift = int(round(ref_trace.y_of_x(ref_x) - tr.y_of_x(ref_x)))
        shifts.append(shift)

        aligned = np.roll(data, shift, axis=0)
        aligned_sat = np.roll(frame_sat, shift, axis=0)
        total = aligned if total is None else total + aligned
        sat_mask = aligned_sat if sat_mask is None else (sat_mask | aligned_sat)

    return total, shifts, sat_mask


def build_header(sub_frames, shifts):
    header = sub_frames[0].header.copy()
    header["IMAGETYP"] = "light"
    header["EXPTIME"] = float(sum(f.exptime for f in sub_frames))
    header["EXPOSURE"] = header["EXPTIME"]
    header["DARKSUB"] = (True, "dark-subtracted per sub-exposure before coadding")
    header["NCOMBINE"] = (len(sub_frames), "number of sub-exposures coadded")
    header.add_history(f"coadd_light.py: combined {len(sub_frames)} sub-exposures")
    for f, shift in zip(sub_frames, shifts):
        name = os.path.basename(f.path)[:60]
        header.add_history(f"  {name} exptime={f.exptime:.1f} yshift={shift:+d}px")
    return header


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_fits")
    ap.add_argument("inputs", nargs="+", help="raw light sub-exposure FITS files")
    ap.add_argument("--dark-dir", default=None,
                     help="directory to search for dark/mdark frames "
                          "(default: directory of the first input file)")
    ap.add_argument("--saturation-adu", type=float, default=65535,
                     help="raw ADU value (per sub-exposure, pre dark-subtraction) "
                          "at/above which a pixel is flagged saturated (default: "
                          "the 16-bit ADC ceiling)")
    args = ap.parse_args()

    sub_frames = sorted((frames.classify(p) for p in args.inputs), key=lambda f: f.mjd)

    dark_dir = args.dark_dir or os.path.dirname(os.path.abspath(args.inputs[0]))
    dark_paths = sorted(glob.glob(os.path.join(dark_dir, "*.fits")))
    dark_candidates = frames.classify_all(dark_paths)
    mdark_frames = [f for f in dark_candidates if f.role == "mdark"]
    dark_frames = [f for f in dark_candidates if f.role == "dark"]

    total, shifts, sat_mask = coadd(sub_frames, mdark_frames, dark_frames,
                                    saturation_adu=args.saturation_adu)
    header = build_header(sub_frames, shifts)

    primary = fits.PrimaryHDU(data=total.astype(np.float32), header=header)
    sat_hdu = fits.ImageHDU(data=sat_mask.astype(np.uint8), name="SATMASK")
    fits.HDUList([primary, sat_hdu]).writeto(args.out_fits, overwrite=True)
    print(f"wrote {args.out_fits}: {len(sub_frames)} frames, "
          f"total exptime {header['EXPTIME']:.1f}s, "
          f"y-shifts {shifts}, {int(sat_mask.sum())} saturated pixels (any sub-exposure)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
